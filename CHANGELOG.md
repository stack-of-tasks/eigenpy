# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [3.12.0] - 2025-08-12

### Added

- Add additional decompositions and solvers from Eigen ([#571](https://github.com/stack-of-tasks/eigenpy/pull/571))

### Added

- Docker images `ghcr.io/stack-of-tasks/eigenpy` ([#575](https://github.com/stack-of-tasks/eigenpy/pull/575))

### Changed

- Change the default branch to `devel` ([#547](https://github.com/stack-of-tasks/eigenpy/pull/547))

## [3.11.0] - 2025-04-25

### Added

- Add user-defined literal ""_a for bp::arg ([#545](https://github.com/stack-of-tasks/eigenpy/pull/545))

### Fixed

- Fix handling of non sorted sparse matrix ([#538](https://github.com/stack-of-tasks/eigenpy/pull/538))

### Changed

- Update clang-format standard to C++11, reformat code

## [3.10.3] - 2025-02-11

### Added
- Add `BUILDING_ROS2_PACKAGE` to generate ament specific file ([#530](https://github.com/stack-of-tasks/eigenpy/pull/530))

### Changed
- Modernize doxygen documentation ([#533](https://github.com/stack-of-tasks/eigenpy/pull/533))

## [3.10.2] - 2025-01-13

### Fixed

- Fix Python library linkage for Debug build on Windows ([#514](https://github.com/stack-of-tasks/eigenpy/pull/514))
- Fix np.ones when dtype is a custom user type ([#525](https://github.com/stack-of-tasks/eigenpy/pull/525))

## [3.10.1] - 2024-10-30

### Added

- Add Pixi support ([#444](https://github.com/stack-of-tasks/eigenpy/pull/444))

### Fixed

- Don't use C++14 feature ([#510](https://github.com/stack-of-tasks/eigenpy/pull/510))
- Add inline to `deprecationTypeToPyObj` definition to avoid linking error ([#512](https://github.com/stack-of-tasks/eigenpy/pull/512))

## [3.10.0] - 2024-09-26

### Added

- `GenericMapPythonVisitor`/`StdMapPythonVisitor` can now take an extra visitor argument in the `expose()` method, similar to `StdVectorPythonVisitor`

### Changed

- Move `GenericMapPythonVisitor` to its own header `eigenpy/map.hpp`
- Rename `overload_base_get_item_for_std_map` to `overload_base_get_item_for_map`, move out of `eigenpy::details` namespace
- Move `EmptyPythonVisitor` to new header `eigenpy/utils/empty-visitor.hpp`

## [3.9.1] - 2024-09-19

### Added

- Add test returning reference of std::pair ([#503](https://github.com/stack-of-tasks/eigenpy/pull/503))
- Add more general visitor `GenericMapPythonVisitor` for map types test `boost::unordered_map<std::string, int>` ([#504](https://github.com/stack-of-tasks/eigenpy/pull/504))
- Support for non-[default-contructible](https://en.cppreference.com/w/cpp/named_req/DefaultConstructible) types in map types ([#504](https://github.com/stack-of-tasks/eigenpy/pull/504))
- Add type_info helpers ([#502](https://github.com/stack-of-tasks/eigenpy/pull/502))
- Add NumPy 2 support ([#496](https://github.com/stack-of-tasks/eigenpy/pull/496))

### Changed

- Move `StdMapPythonVisitor` out of `eigenpy::python` namespace, which was a mistake ([#504](https://github.com/stack-of-tasks/eigenpy/pull/504))

## [3.9.0] - 2024-08-31

### Changed
- The `exposeStdVectorEigenSpecificType()` template function now takes the vector allocator as a template parameter ([#500](https://github.com/stack-of-tasks/eigenpy/pull/500))

### Added
- Add bp::dist to std::map converter ([#499](https://github.com/stack-of-tasks/eigenpy/pull/499))

## [3.8.2] - 2024-08-26

### Fixed
- Fix function signature on Windows ([#494](https://github.com/stack-of-tasks/eigenpy/pull/494))

## [3.8.1] - 2024-08-25

### Fixed
- Fix compatibility issue with NumPy 2.x on Windows ([#492](https://github.com/stack-of-tasks/eigenpy/pull/492))

## [3.8.0] - 2024-08-14

### Added
- Add compatibility with jrl-cmakemodules workspace ([#485](https://github.com/stack-of-tasks/eigenpy/pull/485))
- Remove support of Python 3.7 ([#490](https://github.com/stack-of-tasks/eigenpy/pull/490))

### Fixed
- Remove CMake CMP0167 warnings ([#487](https://github.com/stack-of-tasks/eigenpy/pull/487))
- Fix compilation error on armhf ([#488](https://github.com/stack-of-tasks/eigenpy/pull/488))

## [3.7.0] - 2024-06-11

### Added
- Added id() helper to retrieve unique object identifier in Python ([#477](https://github.com/stack-of-tasks/eigenpy/pull/477))
- Expose QR solvers ([#478](https://github.com/stack-of-tasks/eigenpy/pull/478))

## [3.6.0] - 2024-06-05

### Added
- Added a deprecation call policy shortcut ([#466](https://github.com/stack-of-tasks/eigenpy/pull/466))

### Fixed
- Fix register_symbolic_link_to_registered_type() for multiple successive registrations ([#471](https://github.com/stack-of-tasks/eigenpy/pull/471))

## [3.5.1] - 2024-04-25

### Fixed
- Allow EigenToPy/EigenFromPy specialization with CL compiler ([#462](https://github.com/stack-of-tasks/eigenpy/pull/462))
- Fix missing include for boost >= 1.85  ([#464](https://github.com/stack-of-tasks/eigenpy/pull/464))

## [3.5.0] - 2024-04-14

### Added
- Allow use of installed JRL-cmakemodule ([#446](https://github.com/stack-of-tasks/eigenpy/pull/446)
- Support of Numpy 2.0.0b1 ([#448](https://github.com/stack-of-tasks/eigenpy/pull/448))
- Support new primitive type (char, int8_t, uint8_t, int16_t, uint16_t, uint32_t, uint64_t) ([#455]()https://github.com/stack-of-tasks/eigenpy/pull/455)
- Support conversion between signed <-> unsigned integers ([#455](https://github.com/stack-of-tasks/eigenpy/pull/455))
- Support conversion between complex numbers ([#455](https://github.com/stack-of-tasks/eigenpy/pull/455))

### Fixed
- Fix unit test build in C++11 ([#442](https://github.com/stack-of-tasks/eigenpy/pull/442))
- Fix unit test function signature [#443](https://github.com/stack-of-tasks/eigenpy/pull/443))
- Fix CMake export ([#446](https://github.com/stack-of-tasks/eigenpy/pull/446)
- Fix `int` management on Windows ([#455](https://github.com/stack-of-tasks/eigenpy/pull/455))
- Fix `long long` management on Mac ([#455](https://github.com/stack-of-tasks/eigenpy/pull/455))
- Allow to run test in the build directory on Windows ([#457](https://github.com/stack-of-tasks/eigenpy/pull/457))

### Removed
- Remove casting when converting from Eigen scalar to Numpy scalar.
  This should not remove any functionality since Numpy array are created from the Eigen scalar type
  ([#455](https://github.com/stack-of-tasks/eigenpy/pull/455))

## [3.4.0] - 2024-02-26

### Added
- Support for `Eigen::SparseMatrix` types ([#426](https://github.com/stack-of-tasks/eigenpy/pull/426))
- Support for `boost::variant` types with `VariantConverter` ([#430](https://github.com/stack-of-tasks/eigenpy/pull/430))
- Support for `std::variant` types with `VariantConverter` ([#431](https://github.com/stack-of-tasks/eigenpy/pull/431))
- Support for `std::unique_ptr` as a return types with `StdUniquePtrCallPolicies` and `boost::python::default_call_policies` ([#433](https://github.com/stack-of-tasks/eigenpy/pull/433))
- Support for `std::unique_ptr` as an internal reference with `ReturnInternalStdUniquePtr` ([#433](https://github.com/stack-of-tasks/eigenpy/pull/433))
- Support for `Eigen::Simplicial{LLT,LDLT}` and `Eigen::Cholmod{Simplicial,Supernodal}{LLT,LDLT}` Cholesky de compositions ([#438](https://github.com/stack-of-tasks/eigenpy/pull/438))
- Switch to ruff for lints, format, and import sort ([#441](https://github.com/stack-of-tasks/eigenpy/pull/441))

### Fixed
- Fix the issue of missing exposition of Eigen types with __int64 scalar type ([#426](https://github.com/stack-of-tasks/eigenpy/pull/426))
- Fix namespace use in unittest/std_pair.cpp ([#429](https://github.com/stack-of-tasks/eigenpy/pull/429))
- Fix case of zero-size sparse matrices ([#437](https://github.com/stack-of-tasks/eigenpy/pull/437))

## [3.3.0] - 2024-01-23

### Fixed
- Fix potential memory leak when returning a list from an `std::vector` or an `std::array` ([423](https://github.com/stack-of-tasks/eigenpy/pull/423))

## [3.2.0] - 2023-12-12

### Added
- Support for C++11 `std::array` types ([#412](https://github.com/stack-of-tasks/pull/412))
- Support for `std::pair` types ([#417](https://github.com/stack-of-tasks/pull/417))

## [3.1.4] - 2023-11-27

### Added
- Add new helper functions to check Tensor support

### Fixed
- Fix stub generation on Windows

## [3.1.3] - 2023-11-09

### Fixed
- Install `include/eigenpy/registration_class.hpp`

## [3.1.2] - 2023-11-09

### Added
- Support Python 3.12 ([#391](https://github.com/stack-of-tasks/eigenpy/pull/391))

### Fixed
- Add method to `std::vector<MatrixXX>` binding even if another library had registered it ([#393](https://github.com/stack-of-tasks/eigenpy/pull/393))

### Changed
- CMake minimal version is now 3.10 ([#388](https://github.com/stack-of-tasks/eigenpy/pull/388))

## [3.1.1] - 2023-07-31

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/374
* Fix ROS CI by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/384
* Extended copyable visitor by [@cmastalli](https://github.com/cmastalli) in https://github.com/stack-of-tasks/eigenpy/pull/383
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/382

## [3.1.0] - 2023-06-01

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/362
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/363
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/366
* WIP: Expose boost::none_t type (and std::nullopt_t when available) by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/367
* optional: check registration of none type and optional type before exposing converter by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/368
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/371
* Sync submodule CMake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/373

## [3.0.0] - 2023-04-22

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/345
* Clean use of namespace bp:: by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/346
* Add full support of Eigen::Tensor by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/348
* Enable scalar template specialization by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/349
* Tests: add user-struct test by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/350
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/351
* CMake: have python stubs target depend on pywrap target by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/352
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/354
* fix INSTALL_RPATH on ROS & OSX by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/355
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/356
* Add util to expose optional types by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/357
* Fix CI + sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/358
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/359
* Add example showing how to bind virtual classes, passing to overrides… by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/360
* Remove support of numpy.matrix class by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/361

## [2.9.2] - 2023-02-01

### What's Changed
* Fix for Python 3.6 by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/344

## [2.9.1] - 2023-01-31

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/340
* test python 2/3 ubuntu 18/20/22 by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/341
* Add and expose printEigenVersion by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/342
* Fix issue with Boost.Python < 1.71 by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/343

## [2.9.0] - 2023-01-09

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/333
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/334
* Move and update license by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/335
* Add full support of vectorization by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/336
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/338
* Simplify alignment procedure by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/337
* Remove deprecated EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/339

## [2.8.1] - 2022-12-07

### What's Changed
* Fix handling of Numpy blocks on vector types by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/332

## [2.8.0] - 2022-12-05

### What's Changed
* Modify stride assertion in `numpy-map.hpp` to be valid for empty vector by [@acmiyaguchi](https://github.com/acmiyaguchi) in https://github.com/stack-of-tasks/eigenpy/pull/321
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/323
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/326
* Fix potential issues related to Boost.Python >= 1.80 by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/327
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/328
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/329
* Copy std-vector and std-map from Pinocchio by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/325

### New Contributors
* [@acmiyaguchi](https://github.com/acmiyaguchi) made their first contribution in https://github.com/stack-of-tasks/eigenpy/pull/321

## [2.7.14] - 2022-09-11

### What's Changed
* Fix Boost.Python export by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/319

## [2.7.13] - 2022-09-08

### What's Changed
* Fix packaging by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/311
* add relocatable test by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/313
* [CI] Always run PRERELEASE tests by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/307
* Test https://github.com/jrl-umi3218/jrl-cmakemodules/pull/547 by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/314
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/315
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/317
* Fix flake8 errors by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/318

## [2.7.12] - 2022-08-24

### What's Changed
* Sync submodule CMake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/306
* define HAVE_SNPRINTF for windows, fix #309 by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/310
* export SEARCH_FOR_BOOST_PYTHON by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/308

## [2.7.11] - 2022-08-11

### What's Changed
* CMake: typo by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/301
* FIx issue related to Python 3.11 by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/304
* Sync submodule Cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/305

## [2.7.10] - 2022-07-27

### What's Changed
* pre-commit / cmake-format by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/300
* ci: test packaging by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/298

## [2.7.9] - 2022-07-27

### What's Changed
* Enhance CI by testing hpp-fcl by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/296
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/297

## [2.7.8] - 2022-07-24

### What's Changed
* Fix cross compilation by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/293
* Fix python numpy linking error by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/295

## [2.7.7] - 2022-07-19

### What's Changed
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/287
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/288
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/289
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/290
* cmake: relocatable package for recent CMake versions by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/291
* ROS2/Colcon integration for AMENT_PREFIX_PATH and PYTHONPATH by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/292

## [2.7.6] - 2022-05-22

### What's Changed
* Allow template specialization of getitem by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/286

## [2.7.5] - 2022-05-20

### What's Changed
* Fix for Refs to dynamic (1, N) blocks by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/284
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/285

## [2.7.4] - 2022-05-06

This release fixes a major bug related to Eigen::Ref when using Row Major matrices.

### What's Changed
* unit tests: fix super() for python 2 by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/281
* test/eigen_ref: test using Eigen::Ref as data member by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/282
* Fix RowMajor case for Eigen::Ref by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/283

## [2.7.3] - 2022-05-02

### What's Changed
* numpy: avoid deprecated header by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/280

## [2.7.2] - 2022-04-22

### What's Changed
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/276
* ci: autoupdate devel instead of master by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/277
* [pre-commit.ci] pre-commit autoupdate by [@pre-commit-ci](https://github.com/pre-commit-ci) in https://github.com/stack-of-tasks/eigenpy/pull/278
* Test: modify a matrix block through Python subclass by [@ManifoldFR](https://github.com/ManifoldFR) in https://github.com/stack-of-tasks/eigenpy/pull/279

### New Contributors
* [@pre-commit-ci](https://github.com/pre-commit-ci) made their first contribution in https://github.com/stack-of-tasks/eigenpy/pull/276
* [@ManifoldFR](https://github.com/ManifoldFR) made their first contribution in https://github.com/stack-of-tasks/eigenpy/pull/279

## [2.7.1] - 2022-04-09

### What's Changed
* Expose const Eigen::Ref<const ... + update pre-commit + correctly export the project version by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/275

## [2.7.0] - 2022-04-02

### What's Changed
* Export name types by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/273
* Apply pre-commit on all the project by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/274

## [2.6.11] - 2022-02-25

### What's Changed
* Fix quaternion initialisation segfault under -march=native by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/267
* Fix Quaternion constructor by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/269
* Fix memory issue with Quaternion::normalized by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/271
* Remove useless std::cout by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/272

## [2.6.10] - 2022-02-02

This new release enhances portability for ROS2 and fixes issues with Eigen::RowMajor matrix types.

### What's Changed
* Prepare ROS2 release by [@wxmerkt](https://github.com/wxmerkt) in https://github.com/stack-of-tasks/eigenpy/pull/264
* Fix RowMajor issues by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/266

## [2.6.9] - 2021-10-29

Mostly a maintenance release that enhances packaging support and provides fixes for Windows.

### What's Changed
* Fix stubgen by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/260
* Remove useless setup by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/261
* FetchContent on missing submodule by [@nim65s](https://github.com/nim65s) in https://github.com/stack-of-tasks/eigenpy/pull/262
* Sync submodule cmake by [@jcarpent](https://github.com/jcarpent) in https://github.com/stack-of-tasks/eigenpy/pull/263

## [2.6.8] - 2021-09-05

This new release:
- enhances the compatibility with Boost >= 1.77
- fixes Python doc
- adds the support of Python stubs

## [2.6.7] - 2021-08-19

This new release provides additional support in user-type for extracting class objects exposed through boost python.

## [2.6.6] - 2021-08-13

This new release provides:
- extended support of Custom types with Numpy
- extended support of Eigen::{LLT,LDLT} for matrix solution
- MINRES solver support

## [2.6.5] - 2021-07-30

This new release extends the support of custom types registration within NumPy arrays.

## [2.6.4] - 2021-05-25

This new release fixes some bugs when compiling with -march=native.
It also fixes a bug of include orders.

## [2.6.3] - 2021-04-16

This new release enhances the support of EigenPy with Numpy >= 1.2.0

## [2.6.2] - 2021-03-28

Add support of boolean matrices.

## [2.6.1] - 2021-01-20

This new release fixes a bug related to Quaternion initialization in Python.

## [2.6.0] - 2021-01-04

This new release provides extended support for arbitrary scalar types.

## [2.5.0] - 2020-08-25

This new release provides a support of Eigen::Ref to Python with shared memory.

## [2.4.4] - 2020-08-18

This new release fixes a bug encountered when trying to expose non-square matrix.
It also improves the packaging support on Conda.

## [2.4.3] - 2020-07-20

This new release fixes some packaging issues and removes some CMake warnings.

## [2.4.2] - 2020-07-17

This new release provides extended support to NumPy 1.19 and more.

## [2.4.1] - 2020-06-09

This new release improves the packaging of the project by removing any dependency to pkg-config when searching for dependencies. In addition, the Windows compatibility is enhanced.

## [2.4.0] - 2020-05-25

This new release enables:
- the exposition of user types inside NumPy
- a enhance compatibility with Windows

## [2.3.2] - 2020-04-23

This new release enforces the compatibility with:
- ROS and Python3
- Ubuntu 20.04
- Eigen >= 3.3.90

## [2.3.1] - 2020-04-08

This new release fixes some packaging issues for OS X systems.

## [2.3.0] - 2020-04-03

This new release comes with new features:
- the Geometry module has now full shared memory with Python

and comes with additional bug fixes:
- the project can still be used with former CMake style
- the compilations issues with Boost 1.71 has been fixed
- on OS X systems, the project can be relocated

## [2.2.2] - 2020-03-30

This release fixes a packaging with former versions of ROS.

## [2.2.1] - 2020-03-27

This new release of EigenPy:
- removes boring compilation warnings in C++11
- fixes a bug in the Python documentation of the Geometry module
- fixes the CMake export of the project with respect to ROS

## [2.2.0] - 2020-03-18

This new release fully introduces the support of sharing of memory from Eigen to Numpy. This feature can be disabled thanks to "eigenpy.sharedMemory(False)" to recover the previous behavior.

In addition, `eigenpy::Ref`has been removed, as `Eigen::Ref` is now fully supported.

## [2.1.2] - 2020-02-28

This release fixes a packaging bug with older version of CMake.

## [2.1.1] - 2020-02-27

This new release fixes the support with Windows systems and fixes some bugs with the CMake packaging of the project.

## [2.1.0] - 2020-02-25

This new release adds the full support for:
- sharing the memory between Eigen and Numpy
- provides a complete way of exposing Eigen::Ref (with allocation only when needed)
- improves the whole efficiency of the code

## [2.0.3] - 2020-02-20

This new release fixes:
- improve the search of Python
- fix a potential memory leak when using Eigen::MatrixBase objects

## [2.0.2] - 2020-02-06

This new release provides some additional supports on Eigen::Matrix to np.array conversion. It also reduces the memory overload when compiling EigenPy.

## [2.0.1] - 2020-01-31

This new release improves the compatibility of EigenPy with Win32 systems.

## [2.0.0] - 2020-01-30

This is the new release of EigenPy, which some new features:

- full support of np.array which becomes the default conversion format. The end-user is still able to switch to numpy.matrix by calling eigenpy.switchToNumpyMatrix()
- we provide a full exposition of the Eigen decompositions (LLT, LDLT, EigenSolvers, etc.)
- fixes have been done that were appearing in some very particular cases

## [1.6.13] - 2020-01-10

This new release uniformizes the build of unitary tests and also provides a new function to set the random seed of the std random generator.

## [1.6.12] - 2019-12-10

This release fixes the compatibility on Win32 systems and provides a fix for a bug introduced in 1.6.10.

## [1.6.11] - 2019-12-09

This new release fixes an important bug introduced in 1.6.10.

## [1.6.10] - 2019-12-09

This new release:
- makes the project fully compatible with CMake package policy
- improves the documentation of the Python bindings
- fixes the convertibility of Eigen base classes

## [1.6.9] - 2019-11-25

Missing update of the package.xml file.

## [1.6.8] - 2019-11-25

This new release improves the compatibility version between numpy.array and Eigen::Matrix conversions. It also fixes some compilations issues on Win32 systems.

## [1.6.7] - 2019-11-15

This new release fixes some compilations of EigenPy on the ROS build farm.
Thanks to [@wxmerkt](https://github.com/wxmerkt] for the fix.

## [1.6.6] - 2019-11-13

This new patched release improves the packaging of the project with respect to external forges.

## [1.6.5] - 2019-11-08

This new release fixes the export of the Eigen project for CMake targets relying on EigenPy.
It also provides some fixes aroung Geometry classes.

## [1.6.4] - 2019-11-07

This new release improves the packaging of the project with respect to ROS and also fixes some constructors issues with respect to the Quaternion class.

## [1.6.3] - 2019-10-29

This release fix issues introduced recently in the packaging which was preventing the project to correctly export the main library.

## [1.6.2] - 2019-10-24

This is a maintenance release that improves the packaging for Windows and ROS. It also fixes some bugs in the bindings of Quaternion.

## [1.6.1] - 2019-10-16

This new release provides default exposition of Eigen vector and matrix which are common. On the packaging side, the project is now fully compatible with the new CMake rules for defining and calling the `PROJECT` master function.

## [1.6.0] - 2019-09-19

This new release makes a step towards removing the conversion of `Eigen::Matrix` objects into `numpy.matrix` objects. This new release now warms by default, saying that you need to make an explicit choice between `numpy.matrix` and `numpy.array`.

Future major releases will enforce the default case with `numpy.array`.

## [1.5.8] - 2019-09-09

This new release mostly removes a useless print when loading a converter.

## [1.5.7] - 2019-07-19

This new release adds the support of Windows OS (thanks to [@seanyen](https://github.com/seanyen]).
It also provides some fixes with respect to recent versions of Boost.Python (>= 1.70.0).

## [1.5.6] - 2019-07-16

This release fixes some compatibility issues with anaconda.

## [1.5.5] - 2019-07-15

This new release improves the packaging with respect to Boost >= 1.70.0.
This is needed for complete integration inside Anaconda.

## [1.5.4] - 2019-07-13

This new release mostly improves the packaging of the project, also with ROS.

## [1.5.3] - 2019-06-28

This new release fixes a serious bug with duplicates symbols.

## [1.5.2] - 2019-06-26

This new release fixes two important bugs:
- a type already registered by another library will have a symbolic link in the current scope
- when using the array convention, the Eigen vector objects become flatten numpy.array

## [1.5.1] - 2019-04-16

This new release fixes a serious bug related to Python3 which was occurring when exiting the interpreter.
This new release makes also official the new BSD license.

## [1.5.0] - 2018-10-29

This new release allows now to support the conversion from C++ to Python either as numpy.matrix or as numpy.array.

The use can switch between both features by calling in Python
```
eigenpy.switchToNumpyArray()
```
to convert Eigen::Matrix to numpy.array. Or
```
eigenpy.switchToNumpyMatrix()
```
to select conversion from Eigen::Matrix to numpy.matrix.

Thanks to [@jviereck](https://github.com/jviereck] for raising the lost of performances induced by numpy.matrix.

## [1.4.5] - 2018-08-29

Fix compatibility with Eigen 3.3.5

## [1.4.4] - 2018-07-19

This is a minor release.
It allows to register Eigen::MatrixBase\<MatType\> from Python to Eigen converters.

## [1.4.3] - 2018-05-12

This release fixes several compatibility issues at the Cmake level for both Python 3 and recent versions of Boost >= 1.67.0.

## [1.4.2] - 2018-05-02

This new release allows the binding of Eigen::Ref, facilitates the conversion between compatible scalars (eigenpy won't warn if the numpy matrix has integer scalar type).
This new release also improves the compatibility with Python 3.x.

## [1.4.1] - 2018-02-26

This is mostly a maintenance release where some fixes have been done with respect to BSD systems.
We also fixed an issue concerning the catching of exceptions: they are now verbose.

## [1.4.0] - 2018-01-14

This new release introduces an independent eigenpy library that exposes Eigen solvers in Python.
It also allows to use directly NumPy memory through famous Eigen::Ref that are just memory mapping.

## [1.3.3] - 2017-06-09

It is mostly a maintenance release with suppression of warnings and use of correct types.

## [1.3.2] - 2016-11-21

### Summary

This new release allows the check of registration of any class. This allows to not define twice the same symbols. Thanks to the macro defined in memory.hpp, one can now relies on aligned vector and matrices. The unaligned equivalent type is no more required.
This release also improves the exposed API of Quaternions and AngleAxis Eigen classes. The install of documentation can be avoided by setting cmake option INSTALL_DOCUMENTATION to OFF.

### Bug Fix
- the Python function _import_array() must be called first before any class to PyArray functions.

## [1.3.1] - 2016-09-23

### Summary

This new release fixes several bugs encounter in the linkage of EigenPy with other libs integrating Python interpreter. It also improve the packaging of the module by removing the dependencies to the Python lib and making Boost Python defined with undefined symbols (useful for OS X - the library is no more static on OS X).

### Bug Fix
- UnalignedEquivalent struct take now an optional Scalar template
- PyMatrixType is now a singleton, avoiding bad initialisation during dynamic loading of shared lib.

## [1.3.0] - 2016-02-03

### Summary

Alignment of Eigen and Numpy objects is now properly handled.
One can now easily expose C++ struct containing Eigen objects in Python avoiding any unnecessary conversion and data are aligned in memory.

### Bug Fix
- Fix issue in the creation of row and column vectors.
- The library has to be static under OS X to properly expose symbols.

## [1.2.0] - 2014-11-13

## [1.1.0] - 2014-09-16

## [1.0.1] - 2014-07-18

## [1.0.0] - 2014-07-18

[Unreleased]: https://github.com/stack-of-tasks/eigenpy/compare/v3.12.0...HEAD
[3.12.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.11.0...v3.12.0
[3.11.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.10.3...v3.11.0
[3.10.3]: https://github.com/stack-of-tasks/eigenpy/compare/v3.10.2...v3.10.3
[3.10.2]: https://github.com/stack-of-tasks/eigenpy/compare/v3.10.1...v3.10.2
[3.10.1]: https://github.com/stack-of-tasks/eigenpy/compare/v3.10.0...v3.10.1
[3.10.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.9.1...v3.10.0
[3.9.1]: https://github.com/stack-of-tasks/eigenpy/compare/v3.9.0...v3.9.1
[3.9.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.8.2...v3.9.0
[3.8.2]: https://github.com/stack-of-tasks/eigenpy/compare/v3.8.1...v3.8.2
[3.8.1]: https://github.com/stack-of-tasks/eigenpy/compare/v3.8.0...v3.8.1
[3.8.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.7.0...v3.8.0
[3.7.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.6.0...v3.7.0
[3.6.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.5.1...v3.6.0
[3.5.1]: https://github.com/stack-of-tasks/eigenpy/compare/v3.5.0...v3.5.1
[3.5.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.4.0...v3.5.0
[3.4.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.3.0...v3.4.0
[3.3.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.2.0...v3.3.0
[3.2.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.1.4...v3.2.0
[3.1.4]: https://github.com/stack-of-tasks/eigenpy/compare/v3.1.3...v3.1.4
[3.1.3]: https://github.com/stack-of-tasks/eigenpy/compare/v3.1.2...v3.1.3
[3.1.2]: https://github.com/stack-of-tasks/eigenpy/compare/v3.1.1...v3.1.2
[3.1.1]: https://github.com/stack-of-tasks/eigenpy/compare/v3.1.0...v3.1.1
[3.1.0]: https://github.com/stack-of-tasks/eigenpy/compare/v3.0.0...v3.1.0
[3.0.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.9.2...v3.0.0
[2.9.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.9.1...v2.9.2
[2.9.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.9.0...v2.9.1
[2.9.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.8.1...v2.9.0
[2.8.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.8.0...v2.8.1
[2.8.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.14...v2.8.0
[2.7.14]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.13...v2.7.14
[2.7.13]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.12...v2.7.13
[2.7.12]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.11...v2.7.12
[2.7.11]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.10...v2.7.11
[2.7.10]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.9...v2.7.10
[2.7.9]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.8...v2.7.9
[2.7.8]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.7...v2.7.8
[2.7.7]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.6...v2.7.7
[2.7.6]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.5...v2.7.6
[2.7.5]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.4...v2.7.5
[2.7.4]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.3...v2.7.4
[2.7.3]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.2...v2.7.3
[2.7.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.1...v2.7.2
[2.7.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.7.0...v2.7.1
[2.7.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.11...v2.7.0
[2.6.11]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.10...v2.6.11
[2.6.10]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.9...v2.6.10
[2.6.9]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.8...v2.6.9
[2.6.8]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.7...v2.6.8
[2.6.7]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.6...v2.6.7
[2.6.6]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.5...v2.6.6
[2.6.5]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.4...v2.6.5
[2.6.4]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.3...v2.6.4
[2.6.3]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.2...v2.6.3
[2.6.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.1...v2.6.2
[2.6.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.6.0...v2.6.1
[2.6.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.5.0...v2.6.0
[2.5.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.4.4...v2.5.0
[2.4.4]: https://github.com/stack-of-tasks/eigenpy/compare/v2.4.3...v2.4.4
[2.4.3]: https://github.com/stack-of-tasks/eigenpy/compare/v2.4.2...v2.4.3
[2.4.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.4.1...v2.4.2
[2.4.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.4.0...v2.4.1
[2.4.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.3.2...v2.4.0
[2.3.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.3.1...v2.3.2
[2.3.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.3.0...v2.3.1
[2.3.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.2.2...v2.3.0
[2.2.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.2.1...v2.2.2
[2.2.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.2.0...v2.2.1
[2.2.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.1.2...v2.2.0
[2.1.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/stack-of-tasks/eigenpy/compare/v2.0.3...v2.1.0
[2.0.3]: https://github.com/stack-of-tasks/eigenpy/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/stack-of-tasks/eigenpy/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/stack-of-tasks/eigenpy/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.1...v2.0.0
[1.6.13]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.1...v1.6.1
[1.6.12]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.1...v1.6.1
[1.6.11]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.1...v1.6.1
[1.6.10]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.9...v1.6.1
[1.6.9]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.8...v1.6.9
[1.6.8]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.7...v1.6.8
[1.6.7]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.6...v1.6.7
[1.6.6]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.5...v1.6.6
[1.6.5]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.4...v1.6.5
[1.6.4]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.3...v1.6.4
[1.6.3]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/stack-of-tasks/eigenpy/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.8...v1.6.0
[1.5.8]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.7...v1.5.8
[1.5.7]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.6...v1.5.7
[1.5.6]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.5...v1.5.6
[1.5.5]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.4...v1.5.5
[1.5.4]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.3...v1.5.4
[1.5.3]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/stack-of-tasks/eigenpy/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.5...v1.5.0
[1.4.5]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.4...v1.4.5
[1.4.4]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.3...v1.4.4
[1.4.3]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.2...v1.4.3
[1.4.2]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/stack-of-tasks/eigenpy/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.3.3...v1.4.0
[1.3.3]: https://github.com/stack-of-tasks/eigenpy/compare/v1.3.2...v1.3.3
[1.3.2]: https://github.com/stack-of-tasks/eigenpy/compare/v1.3.1...v1.3.2
[1.3.1]: https://github.com/stack-of-tasks/eigenpy/compare/v1.3.0...v1.3.1
[1.3.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/stack-of-tasks/eigenpy/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/stack-of-tasks/eigenpy/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/stack-of-tasks/eigenpy/releases/tag/v1.0.0
