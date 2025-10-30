use serde::{Serialize,Deserialize};
pub type Level = (f64,f64);
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook { pub bids: Vec<Level>, pub asks: Vec<Level>, pub ts: i64 }
impl OrderBook {
    pub fn new() -> Self { OrderBook { bids: vec![], asks: vec![], ts: 0 } }
    pub fn mid(&self) -> Option<f64> {
        if !self.bids.is_empty() && !self.asks.is_empty() { Some((self.bids[0].0 + self.asks[0].0)/2.0) } else { None }
    }
    pub fn with_snapshot(&self, bids: Vec<Level>, asks: Vec<Level>, ts: i64) -> Self { OrderBook { bids, asks, ts } }
}
