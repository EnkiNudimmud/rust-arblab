fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile(&[
        "../proto/trading.proto",
        "../proto/meanrev.proto",
        "../proto/regime.proto",
        "../proto/drift.proto",
        "../proto/superspace.proto",
        "../proto/pair_discovery.proto",
    ], &["../proto"])?;
    Ok(())
}
