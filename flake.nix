{
  description = "Bindings between Numpy and Eigen using Boost.Python";

  inputs = {
    flake-parts.url = "github:hercules-ci/flake-parts";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    inputs:
    inputs.flake-parts.lib.mkFlake { inherit inputs; } {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;
      perSystem =
        { pkgs, self', ... }:
        {
          apps.default = {
            type = "app";
            program = pkgs.python3.withPackages (_: [ self'.packages.default ]);
          };
          devShells.default = pkgs.mkShell { inputsFrom = [ self'.packages.default ]; };
          packages = {
            default = self'.packages.eigenpy;
            eigen = pkgs.eigen.overrideAttrs {
              # Apply https://gitlab.com/libeigen/eigen/-/merge_requests/977
              postPatch = ''
                substituteInPlace Eigen/src/SVD/BDCSVD.h \
                  --replace-fail "if (l == 0) {" "if (i >= k && l == 0) {"
              '';
            };
            eigenpy =
              (pkgs.python3Packages.eigenpy.override { inherit (self'.packages) eigen; }).overrideAttrs
                (_: {
                  src = pkgs.lib.fileset.toSource {
                    root = ./.;
                    fileset = pkgs.lib.fileset.unions [
                      ./CMakeLists.txt
                      ./doc
                      ./include
                      ./package.xml
                      ./python
                      ./src
                      ./unittest
                    ];
                  };
                });
          };
        };
    };
}
