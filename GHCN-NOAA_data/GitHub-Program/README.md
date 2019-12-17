Get NOAA GHCN Data
==================

This tool functions as an interface to the extensive weather data in the [NOAA GHCN database](https://www.ncdc.noaa.gov/data-access/land-based-station-data/land-based-datasets/global-historical-climatology-network-ghcn). No manually digging through the FTP server or going through Google Big Query, just directly download your station's weather from the command line.

<p align="center"><img src="https://github.com/aaronpenne/get_noaa_ghcn_data/blob/master/screenshots/usage_mac.gif" alt="Screen capture of program usage."></p>

get_noaa_ghcn_data.py
---------------------
Run this tool from the command line to:
- Search for a Global Historical Climatology Network (GHCN) station ID using plain text
- Download a station's daily weather data in raw .dly format
- Download a station's daily weather data in a reformatted .csv for easy analysis

The format of each station's daily measurements are fixed width files with one row per month per element, and multiple columns for the days:

```
    ------------------------------
    Variable   Columns   Type
    ------------------------------
    ID            1-11   Character
    YEAR         12-15   Integer
    MONTH        16-17   Integer
    ELEMENT      18-21   Character
    VALUE1       22-26   Integer
    MFLAG1       27-27   Character
    QFLAG1       28-28   Character
    SFLAG1       29-29   Character
    VALUE2       30-34   Integer
    MFLAG2       35-35   Character
    QFLAG2       36-36   Character
    SFLAG2       37-37   Character
      .           .          .
      .           .          .
      .           .          .
    VALUE31    262-266   Integer
    MFLAG31    267-267   Character
    QFLAG31    268-268   Character
    SFLAG31    269-269   Character
```

This tool parses all of the station's available data and outputs it with one row per day, and multiple columns for the elements. This is easier to analyze programattically and read, as seen below: 

![Screenshot of reformatted .csv file](https://github.com/aaronpenne/get_noaa_ghcn_data/blob/master/screenshots/csv.png)


Background
----------

> The Global Historical Climatology Network (GHCN) is an integrated database of climate summaries from land surface stations across the globe that have been subjected to a common suite of quality assurance reviews. The data are obtained from more than 20 sources. Some data are more than 175 years old while others are less than an hour old. GHCN is the official archived dataset, and it serves as a replacement product for older NCEI-maintained datasets that are designated for daily temporal resolution

The five core elements (measurements) are:
- PRCP = Precipitation (tenths of mm)
- SNOW = Snowfall (mm)
- SNWD = Snow depth (mm)
- TMAX = Maximum temperature (tenths of degrees C)
- TMIN = Minimum temperature (tenths of degrees C)

For a full list of possible elements (measurements) see the [codebook](https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt).

The five core elements which are recorded by most stations are precipitation, snow, snow depth, max temp, and min temp. There are several other attributes that may or may not be recorded in each station's dataset including: average cloudiness, average daily wind direction, average daily wind speed, water evaporation, time of fastest wind, frozen ground layer depth, water depth gauge heights, evaporation pan water temperature, percent of day that is sunshine, soil temp, soil type, ice thickness, direction of fastest winds, weather type, multiday measurements, and number of days included in multiday measurements. There are also flags indicating measurement details, quality details, and the source agency of the measurement.

Dependencies
------------
- [Python 3](https://www.python.org/)
- [pandas](https://github.com/pandas-dev/pandas)

Developed with Python 3.6 on Windows 10. 


License
-------
[MIT License](https://github.com/aaronpenne/get_noaa_ghcn_data/blob/master/LICENSE.md) © Aaron Penne

Disclaimer: This project is not affiliated with NOAA or GHCN in any way.
