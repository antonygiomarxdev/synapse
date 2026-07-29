fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Compile only spike.proto for the viability spike binary.
    // The main synapse.proto is not yet used in production code.
    prost_build::compile_protos(&["../proto/spike.proto"], &["../proto/"])?;
    Ok(())
}
