fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure()
        .build_server(true)
        .build_client(false)
        .compile(
            &["../proto/trading.proto", "../proto/pair_discovery.proto"],
            &["../proto"]
        )?;
    Ok(())
}
