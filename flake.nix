{
  description = "Bindings between Numpy and Eigen using Boost.Python";

  inputs.gepetto.url = "github:gepetto/nix";

  outputs =
    inputs:
    inputs.gepetto.lib.mkFlakoboros inputs (
      { lib, ... }:
      {
        pyOverrideAttrs.eigenpy = { drv-prev, ... }: {
          cmakeFlags = drv-prev.cmakeFlags ++ [ "-DBUILD_DOCUMENTATION=ON" ];
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./cmake
              ./CMakeLists.txt
              ./doc
              ./include
              ./package.xml
              ./python
              ./src
              ./unittest
            ];
          };
        };
        extends.eigen5 = final: prev: {
          eigen = final.eigen_5;
          pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
            (_python-final: python-prev: {
              scipy = python-prev.scipy.overrideAttrs {
                # broken on linux arm
                doInstallCheck = false;
              };
            })
          ];
        };
      }
    );
}
