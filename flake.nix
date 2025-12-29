{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };

        # ffmpeg 7.x with libsvtav1 encoder (required by lerobot)
        ffmpeg-pkg = pkgs.ffmpeg_7 or pkgs.ffmpeg;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
            ffmpeg-pkg
            uv
            # Required for building PyAV against system ffmpeg
            pkg-config
            cmake
          ];

          shellHook = ''
            echo "ffmpeg version: $(ffmpeg -version | head -n 1)"

            # Ensure PyAV finds nix-installed ffmpeg headers/libs
            export PKG_CONFIG_PATH="${ffmpeg-pkg}/lib/pkgconfig:$PKG_CONFIG_PATH"
          '';
        };
      }
    );
}

