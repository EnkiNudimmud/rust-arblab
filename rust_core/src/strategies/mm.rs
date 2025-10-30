pub struct MMQuote{pub bid_px:f64,pub ask_px:f64,pub bid_size:f64,pub ask_size:f64}
pub fn imbalance_quote(bids:&[(f64,f64)], asks:&[(f64,f64)], spread:f64, skew:f64)->MMQuote{
    let best_bid=bids.first().map(|x|x.0).unwrap_or(0.0);
    let best_ask=asks.first().map(|x|x.0).unwrap_or(0.0);
    let mid=(best_bid+best_ask)/2.0;
    let bid=mid - spread*(1.0+skew);
    let ask=mid + spread*(1.0-skew);
    MMQuote{bid_px:bid,ask_px:ask,bid_size:1.0,ask_size:1.0}
}
