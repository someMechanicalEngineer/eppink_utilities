# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## Version 0.1.0 
2025-05-28
### Changed
- Initial upload

## Version 0.2.0 
2025-06-04
### Changed
- Automatic Github pusher

## Version 0.3.0 
2025-06-04
### Changed
- Automatic PYPI pusher
- Automatic version bumper

## Version 0.3.6 
2025-06-04
### Changed
- Fixed tracker, homepage and source links

## Version 0.4.0 
2025-06-06
### Changed
- Added dimless_utils containing the first set of dimensionless numbers for fluid, heat and mass transfer
- Added file_utils containing functions for markdown and csv
- Added generat_utils
- Added plot_utils

## Version 0.4.1 
2025-06-07
### Changed
- Added todo list
- Changed name of some file_utils functions

## Version 0.5.0 
2025-06-08
### Changed
- Now automatically updates FUNCTIONS.md

## Version 0.6.0 
2025-06-08
### Changed
- Automatically update CITATION.cff with regards to year and version bump

## Version 0.6.1 
2025-06-08
### Changed
- test

## Version 0.6.2 
2025-06-11
### Changed
- Added averaging function

## Version 0.6.3 
2025-06-15
### Changed
- Bugfix math_utils.safe_divide

## Version 0.7.0 
2025-06-18
### Changed
- Added derivative_FDM and conduction_radial_numerical functions
## Version 0.7.1 
2025-06-19
### Changed
- bugfix where sister library couldn't be accessed

## Version 0.8.0 
2025-06-19
### Changed
- bugfix where sister library couldn't be accessed

## Version 0.9.0 
2025-06-20
### Changed
- Added functions such as mass_leakrate,  catastrophic_cancellation, data_combine, data_bin, data_column_combine

## Version 0.9.1 
2025-06-20
### Changed
- fixed a bug where the file_utils were trying to open or write to the wrong directory.

## Version 0.9.2 
2025-06-20
### Changed
- bugfix. Removed a print command from csv_open which floods the terminal

## Version 0.10.0 
2025-06-20
### Changed
- Addition of function dataset_splitheaderfromdata
## Version 0.10.1 
2025-06-20
### Changed
- bugfix dataset_combine.

## Version 0.11.0 
2025-06-21
### Changed
- changed csv_filecheck into a more general file_check and added data_remove_columns

## Version 0.11.1 
2025-06-21
### Changed
- derivative_FDM changed to return all variables regularly instead of as a dict.

## Version 0.11.2 
2025-06-21
### Changed
- Updated data_remove_rows to match the functionality of data_remove_columns (i.e., range selection, keep/remove functionality)
## Version 0.11.3 
2025-06-21
### Changed
- renaming value to range in data_remove_rows

## Version 0.11.4 
2025-06-21
### Changed
- data_remove_rows now supports selection of (n,) where n is the minimum and runs to the max.

## Version 0.11.5 
2025-06-21
### Changed
- Dugfix where data_remove row kept the first row using range selection

## Version 0.11.6 
2025-06-21
### Changed
- data_remove_row now also works for datasets with header or without
## Version 0.12.0 
2025-06-22
### Changed
- new heattransfer_utils.convection_analytical, heattransfer_utils.Nusselt_correlations_free and dimless_utils.rayleigh

## Version 0.12.1 
2025-06-23
### Changed
- Changed the returns of conduction_radial_steady_numerical

## Version 0.12.2 
2025-06-23
### Changed
- Bugfix Grashof number. Now takes np.abs(dT)

## Version 0.12.3 
2025-06-23
### Changed
- reverted grashof number bugfix. Problem was that negative values for physical properties were fed into the convection equation. These are nou caught.

## Version 0.12.4 
2025-06-23
### Changed
- Grashof number np.abs(dT) was required after all....

## Version 0.12.5 
2025-06-24
### Changed
- bugfix convection function

## Version 0.12.6 
2025-06-24
### Changed
- bugfix conection function

## Version 0.12.10 
2025-06-24
### Changed
- Problem with convection function was that a negative d caused problems with grashof. Now throws an error with negative d.

## Version 0.12.11 
2025-06-24
### Changed
- Forgot to include d when calling the check positive function
## Version 0.13.0 
2025-06-24
### Changed
- Added sleepmode_prevent and sleepmode_restore

## Version 0.14.0 
2025-06-25
### Changed
- Start of lookuptables. Furthermore addition of cartesian numerical conduction function
