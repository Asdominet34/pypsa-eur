# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Build hydroelectric inflow time-series for each country.

Outputs
-------

- `resources/profile_hydro.nc`:

| Field | Dimensions | Description |
| --- | --- | --- |
| inflow | countries, time | Inflow to the state of charge (in MW), e.g. due to river inflow in hydro reservoir. |

![](img/inflow-ts.png)

![](img/inflow-box.png)
"""

import logging

import country_converter as coco
import geopandas as gpd
import pandas as pd
import xarray as xr
from numpy.polynomial import Polynomial

from scripts._helpers import (
    configure_logging,
    get_snapshots,
    load_cutout,
    set_scenario_config,
)

cc = coco.CountryConverter()


def get_eia_annual_hydro_generation(
    fn: str, countries: list[str], capacities: bool = False
) -> pd.DataFrame:
    # in billion kWh/a = TWh/a
    df = pd.read_csv(fn, skiprows=2, index_col=1, na_values=[" ", "--"]).iloc[1:, 1:]
    df.index = df.index.str.strip()
    df.columns = df.columns.astype(int)

    former_countries = {
        "Former Czechoslovakia": dict(
            countries=["Czechia", "Slovakia"], start=1980, end=1992
        ),
        "Former Serbia and Montenegro": dict(
            countries=["Serbia", "Montenegro", "Kosovo"], start=1992, end=2005
        ),
        "Former Yugoslavia": dict(
            countries=[
                "Slovenia",
                "Croatia",
                "Bosnia and Herzegovina",
                "Serbia",
                "Kosovo",
                "Montenegro",
                "North Macedonia",
            ],
            start=1980,
            end=1991,
        ),
    }

    for k, v in former_countries.items():
        period = [i for i in range(v["start"], v["end"] + 1)]
        ratio = df.loc[v["countries"]].T.dropna().sum()
        ratio /= ratio.sum()
        for country in v["countries"]:
            df.loc[country, period] = df.loc[k, period] * ratio[country]

    baltic_states = ["Latvia", "Estonia", "Lithuania"]
    df.loc[baltic_states] = (
        df.loc[baltic_states].T.fillna(df.loc[baltic_states].mean(axis=1)).T
    )

    df.loc["Germany"] = df.filter(like="Germany", axis=0).sum()
    df = df.loc[~df.index.str.contains("Former")]
    df.drop(["Europe", "Germany, West", "Germany, East"], inplace=True)

    df.index = cc.convert(df.index, to="iso2")
    df.index.name = "countries"

    # convert to MW of MWh/a
    factor = 1e3 if capacities else 1e6
    df = df.T[countries] * factor

    df.ffill(axis=0, inplace=True)

    return df


def correct_eia_stats_by_capacity(
    eia_stats: pd.DataFrame, fn: str, countries: list[str], baseyear: int = 2019
) -> None:
    cap = get_eia_annual_hydro_generation(fn, countries, capacities=True)
    ratio = cap / cap.loc[baseyear]
    eia_stats_corrected = eia_stats / ratio
    to_keep = ["AL", "AT", "CH", "DE", "GB", "NL", "RS", "XK", "RO", "SK"]
    to_correct = eia_stats_corrected.columns.difference(to_keep)
    eia_stats.loc[:, to_correct] = eia_stats_corrected.loc[:, to_correct]


def approximate_missing_eia_stats(
    eia_stats: pd.DataFrame, runoff_fn: str, countries: list[str]
) -> pd.DataFrame:
    runoff = pd.read_csv(runoff_fn, index_col=0, parse_dates=True)[countries]
    runoff = runoff.groupby(runoff.index.year).sum().index

    # fix outliers; exceptional floods in 1977-1979 in ES & PT
    if "ES" in runoff:
        runoff.loc[1978, "ES"] = runoff.loc[1979, "ES"]
    if "PT" in runoff:
        runoff.loc[1978, "PT"] = runoff.loc[1979, "PT"]

    runoff_eia = runoff.loc[eia_stats.index]

    eia_stats_approximated = {}

    for c in countries:
        X = runoff_eia[c]
        Y = eia_stats[c]

        to_predict = runoff.index.difference(eia_stats.index)
        X_pred = runoff.loc[to_predict, c]

        p = Polynomial.fit(X, Y, 1)
        Y_pred = p(X_pred)

        eia_stats_approximated[c] = pd.Series(Y_pred, index=to_predict)

    eia_stats_approximated = pd.DataFrame(eia_stats_approximated)
    return pd.concat([eia_stats, eia_stats_approximated]).sort_index()


logger = logging.getLogger(__name__)


def override_eia_stats(eia_stats: pd.DataFrame, fn: str, countries: list[str]) -> pd.DataFrame:
    """
    Replace the EIA annual hydro generation with observed annual totals per country
    (columns year, country, hydro_generation_mwh), for the countries and years in the file.

    Same as the ENTSO-E override of the GLOFAS-EFAS-hydromodeling fork (pypsa-eur-hydro).
    """
    observed = pd.read_csv(fn)
    missing = {"year", "country", "hydro_generation_mwh"}.difference(observed.columns)
    if missing:
        raise ValueError(f"Missing columns in {fn}: {sorted(missing)}")
    observed = observed[observed.country.isin(countries)].astype({"year": int})
    observed = observed.pivot_table(
        index="year", columns="country", values="hydro_generation_mwh", aggfunc="first"
    )
    for year in observed.index:
        if year not in eia_stats.index:
            eia_stats.loc[year, :] = pd.NA
        common = observed.columns.intersection(eia_stats.columns)
        eia_stats.loc[year, common] = observed.loc[year, common]
    logger.info(
        f"Replaced EIA hydro generation with {fn} for years {observed.index.tolist()}"
    )
    return eia_stats.sort_index()


def build_external_hydro_profile(
    inflow_fns: list[str],
    plants_fn: str,
    countries: list[str],
    time: pd.DatetimeIndex,
) -> xr.Dataset:
    """
    Read plant-level hydro series of an external dataset (macro-hydro-inputs).

    Returns ``inflow`` [MW] for hydro and PHS plants and ``p_max_pu`` for ror
    plants on (time, plant), for the plants of the modelled countries. Inflow
    excludes turbine efficiency: it is applied on discharge.
    """
    plants = pd.read_csv(plants_fn, dtype={"plant_id": str}, index_col="plant_id")
    plants = plants[plants.country.isin(countries)]

    with xr.open_mfdataset(inflow_fns, combine="by_coords") as ds:
        ds = ds.assign_coords(plant=ds.indexes["plant"].astype(str))
        missing_plants = plants.index.difference(ds.indexes["plant"])
        if not missing_plants.empty:
            raise ValueError(
                f"No inflow series for {len(missing_plants)} plants, "
                f"e.g. {list(missing_plants[:10])}"
            )
        missing_time = time.difference(ds.indexes["time"])
        if not missing_time.empty:
            raise ValueError(
                f"{len(missing_time)} snapshots missing in {inflow_fns}, "
                f"e.g. {list(missing_time[:5])} (the files have no 29 February)"
            )
        ds = ds[["inflow", "p_max_pu"]].sel(time=time, plant=plants.index).load()

    is_ror = plants.carrier == "ror"
    for var, needed in (("p_max_pu", is_ror), ("inflow", ~is_ror)):
        values = ds[var].sel(plant=plants.index[needed])
        if values.isnull().any():
            bad = values.plant[values.isnull().any("time")].values[:10]
            raise ValueError(f"NaN in '{var}' of plants {list(bad)}")

    ds["inflow"] = ds["inflow"].clip(min=0.0).astype("float32")
    ds["p_max_pu"] = ds["p_max_pu"].clip(0.0, 1.0).astype("float32")
    logger.info(
        f"External hydro: {len(plants)} plants, inflow "
        f"{float(ds['inflow'].sel(plant=plants.index[~is_ror]).sum()) / 1e6:.1f} TWh "
        f"(hydro and PHS, before efficiency)"
    )
    return ds


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("build_hydro_profile")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    params_hydro = snakemake.params.hydro

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    external = params_hydro.get("external", {})
    if (
        external.get("enable", False)
        and external.get("inflow_and_storage", "external") == "external"
    ):
        build_external_hydro_profile(
            snakemake.input.external_hydro_inflow,
            snakemake.input.external_hydro_plants,
            snakemake.params.countries,
            pd.DatetimeIndex(time),
        ).to_netcdf(
            snakemake.output.profile,
            encoding={v: {"zlib": True, "complevel": 4} for v in ("inflow", "p_max_pu")},
        )
    else:
        cutout = load_cutout(snakemake.input.cutout)

        years_in_time = pd.DatetimeIndex(time).year.unique()
        cutout_time = pd.DatetimeIndex(cutout.coords["time"].values)

        full_years_available = all(
            pd.Timestamp(f"{year}-01-01") in cutout_time
            and pd.Timestamp(f"{year}-12-31") in cutout_time
            for year in years_in_time
        )

        if full_years_available:
            mask = [pd.Timestamp(t).year in years_in_time for t in cutout_time]
            cutout = cutout.sel(time=cutout_time[mask])
        else:
            cutout = cutout.sel(time=time)

        countries = snakemake.params.countries
        country_shapes = (
            gpd.read_file(snakemake.input.country_shapes)
            .set_index("name")["geometry"]
            .reindex(countries)
        )
        country_shapes.index.name = "countries"

        fn = snakemake.input.eia_hydro_generation
        eia_stats = get_eia_annual_hydro_generation(fn, countries)

        config_hydro = snakemake.config["renewable"]["hydro"]

        if config_hydro.get("entsoe_annual_generation"):
            eia_stats = override_eia_stats(
                eia_stats, snakemake.input.entsoe_annual_generation, countries
            )

        if config_hydro.get("eia_correct_by_capacity"):
            fn = snakemake.input.eia_hydro_capacity
            correct_eia_stats_by_capacity(eia_stats, fn, countries)

        if config_hydro.get("eia_approximate_missing"):
            fn = snakemake.input.era5_runoff
            eia_stats = approximate_missing_eia_stats(eia_stats, fn, countries)

        norm_year = config_hydro.get("eia_norm_year")
        missing_years = years_in_time.difference(eia_stats.index)
        if norm_year:
            eia_stats.loc[years_in_time] = eia_stats.loc[norm_year]
        elif missing_years.any():
            for year in missing_years:
                eia_stats.loc[year] = eia_stats.median()

        inflow = cutout.runoff(
            shapes=country_shapes,
            smooth=True,
            lower_threshold_quantile=True,
            normalize_using_yearly=eia_stats,
        )

        if full_years_available:
            inflow = inflow.sel(time=time)

        if "clip_min_inflow" in params_hydro:
            inflow = inflow.where(inflow > params_hydro["clip_min_inflow"], 0)

        inflow.to_netcdf(snakemake.output.profile)
