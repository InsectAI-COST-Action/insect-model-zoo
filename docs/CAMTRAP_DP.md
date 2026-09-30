# Camtrap DP export

[← back to the README](../README.md)

With `--camtrapdp` (or `CAMTRAPDP = True`), the zoo also writes a
[Camtrap DP 1.0.2](https://camtrap-dp.tdwg.org) data package to `output/<detector>+<classifier>/camtrap-dp/`: the
TDWG standard for camera-trap data, read by GBIF, Agouti, camtraptor (R) and others. It holds `datapackage.json`
(metadata) and `deployments.csv`, `media.csv`, `observations.csv`.

- **One deployment** (camera / site) per run, named after the images folder (or `--deployment_id`). Its position comes
  from `--latitude` / `--longitude` (or `CAMTRAPDP_INFO`), else from the photos' GPS. Without a position the package is
  written but is **not valid yet**: the terminal says so, and you can fill in `latitude` / `longitude` in
  `deployments.csv`.
- **Time** of each photo: its EXIF date, with the time zone the camera stored. Most cameras store none: then
  `timezone` in `CAMTRAPDP_INFO` is used (e.g. `"+02:00"` or `"Europe/Copenhagen"`), else this computer's zone, and
  `mediaComments` says it was assumed. Without an EXIF date: the file's modification time, and the deployment is
  marked `timestampIssues`. The deployment's start / end are the first / last photo.
- **One observation per detection** (`observationLevel` = media), with the box as fractions of the image
  (`bboxX, bboxY, bboxWidth, bboxHeight`), `classificationMethod` = machine, `classifiedBy` = the models used and
  `classificationProbability` = the classifier's score (or the detector's when there is no classifier). A photo
  without detections gets one `blank` observation.
- **Scientific names** only: BioCLIP's are already scientific; insectDCT's own class names are converted (e.g.
  *Aranaea* → Araneae, *Birds* → Aves, *Hymenoptera_bees* → Hymenoptera with the comment "bee"), and boxes it calls
  *Vegetation* are left out. Without a classifier: Insecta (insectDCT) or Arthropoda (flat-bug).
- `filePath` links to each photo where it is (`file:///...`): to publish the package, replace them with the photos'
  web addresses (or upload the photos with it, e.g. to Agouti).
- **GBIF:** the package is ready for GBIF's converter (`gbifIngestion` = media-level observations). GBIF also needs a
  contact and a data licence: set `contributor` and `data_license` in `CAMTRAPDP_INFO` (the terminal reminds you).
- **Check `CAMTRAPDP_INFO`** at the top of `main.py` first: project title, your name, and how the camera took photos
  (`timeLapse` or `activityDetection`) and was placed (`targeted`, `opportunistic`, ...). The package is tested
  against the official Camtrap DP schemas.
