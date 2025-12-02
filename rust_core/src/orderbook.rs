use serde::{Serialize, Deserialize};
use std::collections::{BTreeMap, VecDeque};
use std::cmp::Ordering;

// Wrapper for f64 that implements Ord by treating NaN as equal to itself
// and placing it at the end of the ordering
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            // Handle NaN: NaN == NaN for our purposes
            if self.0.is_nan() && other.0.is_nan() {
                Ordering::Equal
            } else if self.0.is_nan() {
                Ordering::Greater  // NaN is "greater" than everything
            } else {
                Ordering::Less
            }
        })
    }
}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl From<f64> for OrderedFloat {
    fn from(f: f64) -> Self {
        OrderedFloat(f)
    }
}

impl From<OrderedFloat> for f64 {
    fn from(of: OrderedFloat) -> Self {
        of.0
    }
}

pub type Price = f64;
pub type Qty = f64;
pub type Ts = i64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub price: Price,
    pub qty: Qty,
    pub ts: Ts,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookSide {
    // price -> FIFO queue of (order-id, qty, ts)
    pub levels: BTreeMap<OrderedFloat, VecDeque<Order>>,
    pub is_bid: bool,
}

impl OrderBookSide {
    pub fn new(is_bid: bool) -> Self {
        Self { levels: BTreeMap::new(), is_bid }
    }
    pub fn best_price(&self) -> Option<Price> {
        if self.is_bid {
            self.levels.keys().rev().next().map(|of| of.0)
        } else {
            self.levels.keys().next().map(|of| of.0)
        }
    }
    pub fn total_qty(&self) -> Qty {
        self.levels.values().map(|q| q.iter().map(|o| o.qty).sum::<Qty>()).sum()
    }
    pub fn add_limit(&mut self, id: u64, price: Price, qty: Qty, ts: Ts) {
        let q = self.levels.entry(OrderedFloat(price)).or_insert_with(VecDeque::new);
        q.push_back(Order { id, price, qty, ts });
    }
    pub fn consume_at_price(&mut self, price: Price, mut qty: Qty) -> (Qty, f64, Vec<(u64, Qty, Price)>) {
        // returns (filled_qty, cost, fills)
        let mut filled = 0.0;
        let mut cost = 0.0;
        let mut fills = Vec::new();
        if let Some(queue) = self.levels.get_mut(&OrderedFloat(price)) {
            while qty > 0.0 {
                if let Some(o) = queue.front().cloned() {
                    let take = Qty::min(o.qty, qty);
                    filled += take;
                    cost += take * price;
                    fills.push((o.id, take, price));
                    // mutate front
                    let front = queue.front_mut().unwrap();
                    front.qty -= take;
                    qty -= take;
                    if front.qty <= 1e-12 {
                        queue.pop_front();
                    }
                } else { break; }
            }
            if queue.is_empty() { self.levels.remove(&OrderedFloat(price)); }
        }
        (filled, cost, fills)
    }
    pub fn price_iter<'a>(&'a self) -> Box<dyn Iterator<Item=Price> + 'a> {
        if self.is_bid {
            Box::new(self.levels.keys().rev().map(|of| of.0))
        } else {
            Box::new(self.levels.keys().map(|of| of.0))
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub bids: OrderBookSide,
    pub asks: OrderBookSide,
    pub ts: Ts,
    seq: u64,
}

impl OrderBook {
    pub fn new() -> Self {
        Self { bids: OrderBookSide::new(true), asks: OrderBookSide::new(false), ts: 0, seq: 1 }
    }
    pub fn best_bid(&self) -> Option<Price> { self.bids.best_price() }
    pub fn best_ask(&self) -> Option<Price> { self.asks.best_price() }
    pub fn mid(&self) -> Option<Price> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((a+b)/2.0),
            _ => None
        }
    }
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a-b),
            _ => None
        }
    }
    pub fn apply_snapshot(&mut self, bids: &[(Price, Qty)], asks: &[(Price, Qty)], ts: Ts) {
        self.bids.levels.clear(); self.asks.levels.clear();
        for (p,q) in bids { 
            if *q>0.0 { 
                let id = self.next_id();
                self.bids.add_limit(id, *p, *q, ts); 
            } 
        }
        for (p,q) in asks { 
            if *q>0.0 { 
                let id = self.next_id();
                self.asks.add_limit(id, *p, *q, ts); 
            } 
        }
        self.ts = ts;
    }
    
    pub fn apply_delta(&mut self, bid_d: &[(Price, Qty)], ask_d: &[(Price, Qty)], ts: Ts) {
        for (p, q) in bid_d {
            if *q <= 0.0 { self.bids.levels.remove(&OrderedFloat(*p)); }
            else { 
                let id = self.next_id();
                self.bids.add_limit(id, *p, *q, ts); 
            }
        }
        for (p, q) in ask_d {
            if *q <= 0.0 { self.asks.levels.remove(&OrderedFloat(*p)); }
            else { 
                let id = self.next_id();
                self.asks.add_limit(id, *p, *q, ts); 
            }
        }
        self.ts = ts;
    }
    fn next_id(&mut self) -> u64 { let id=self.seq; self.seq+=1; id }
}
