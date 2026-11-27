{
  description = "scoria — RDNA4-native LLM inference in Rust with standalone HIP kernels";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      inherit (nixpkgs) lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      perSystem =
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            overlays = [ (import rust-overlay) ];
          };

          # Latest stable. `rust-toolchain.toml` pins an exact release for non-Nix
          # checkouts; the flake is the "always current stable" path.
          rust = pkgs.rust-bin.stable.latest.default.override {
            extensions = [
              "clippy"
              "rustfmt"
              "rust-src"
            ];
          };

          # Device code needs the ROCm LLVM, not stock clang: `--rocm-path` supplies the
          # OCML/OCKL bitcode linked into `-x hip --offload-device-only` output. Optional
          # so a host with /opt/rocm still works without the Nix package.
          # nixpkgs packages ROCm's compiler and runtime but *not* the amdgcn device
          # bitcode (ockl.bc/ocml.bc) that `-x hip --offload-device-only` links device
          # code against, so its clang cannot compile these kernels. The shell and the
          # package therefore use the host's ROCm, which has the bitcode:
          #   build:     nix build --option extra-sandbox-paths /opt/rocm
          #   dev shell: works as-is on a machine with /opt/rocm
          # Set SCORIA_ROCM_PATH / SCORIA_ROCM_DEVICE_LIB in the environment to point
          # somewhere else.
          rocmRoot = "/opt/rocm";

          # scoria links nothing at build time: HIP is dlopen'd at runtime and kernels are
          # compiled to .hsaco by build.rs. So the package needs a compiler, not ROCm libs.
          scoria = pkgs.rustPlatform.buildRustPackage {
            pname = "scoria";
            version = "0.1.0";
            src = self;

            cargoLock.lockFile = ./Cargo.lock;
            cargoBuildFlags = [
              "--package"
              "scoria-cli"
            ];

            nativeBuildInputs = [
              rust
              pkgs.clang
              pkgs.cmake
              pkgs.pkg-config
            ];

            SCORIA_ROCM_PATH = rocmRoot;

            # The test suite runs the GPU gate and disassembles kernels, so it cannot run
            # in the build sandbox. `nix develop` runs it against the real card instead.
            doCheck = false;

            meta = {
              description = "AMD-native LLM inference engine: Rust host, standalone HIP kernels";
              homepage = "https://github.com/risingpresage/scoria";
              license = lib.licenses.mit;
              mainProgram = "scoria";
              platforms = lib.platforms.linux;
            };
          };
        in
        {
          packages = {
            default = scoria;
            inherit scoria;
          };

          apps.default = {
            type = "app";
            program = "${scoria}/bin/scoria";
          };

          devShells.default = pkgs.mkShell {
            packages = [
              rust
              pkgs.clang
              pkgs.cmake
              pkgs.pkg-config
            ];

            SCORIA_ROCM_PATH = rocmRoot;

            shellHook = ''
              echo "scoria: $(rustc --version)"
              if [ -d "${rocmRoot}/lib/llvm/bin" ]; then
                echo "  ROCm:   ${rocmRoot} (kernels will compile)"
              else
                echo "  ROCm:   missing at ${rocmRoot}; set SCORIA_ROCM_PATH to a ROCm install"
              fi
              echo "  verify: ./scripts/check.sh && ./target/release/scoria gate"
            '';
          };

          formatter = pkgs.nixfmt-rfc-style;
        };
    in
    {
      packages = lib.genAttrs systems (s: (perSystem s).packages);
      apps = lib.genAttrs systems (s: (perSystem s).apps);
      devShells = lib.genAttrs systems (s: (perSystem s).devShells);
      formatter = lib.genAttrs systems (s: (perSystem s).formatter);
    };
}
