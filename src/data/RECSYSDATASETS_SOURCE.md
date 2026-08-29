# Dataset sources

This artifact includes the processed item and temporal sequences required by
`./reproduce.sh`. The reproduction command checks their presence and alignment
before every run.

- Beauty and Sports are derived from the Amazon review datasets distributed by
  the public [RecSysDatasets collection](https://github.com/RUCAIBox/RecSysDatasets).
- LastFM is derived from the HetRec 2011 Last.fm 2K timestamped-tag source from
  [GroupLens](https://grouplens.org/datasets/hetrec-2011/), using 1,460-day
  global-start absolute time buckets.
- ML-1M is derived from MovieLens 1M `ratings.dat` from
  [GroupLens](https://grouplens.org/datasets/movielens/1m/), using 90-day
  global-start absolute time buckets.

Raw archives are not redistributed. The checked-in processed files are the
fixed inputs used for the reported experiments.
