pub fn zscore_series(x:&[f64],w:usize)->Vec<f64>{
    if x.len()<w { return Vec::new(); }
    let mut out=Vec::new();
    for i in 0..=x.len()-w {
        let ww=&x[i..i+w];
        let m=ww.iter().sum::<f64>()/(w as f64);
        let v=ww.iter().map(|v|(v-m)*(v-m)).sum::<f64>()/(w as f64);
        let s=v.sqrt().max(1e-9);
        out.push((x[i+w-1]-m)/s);
    }
    out
}
