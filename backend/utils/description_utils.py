from typing import Dict, Any


def build_semantic_description(entry: Dict[str, Any]) -> str:
    """Return a human readable summary of a benchmark."""
    name = entry.get("name", "")
    tags = entry.get("tags", {})
    fundamentals = entry.get("fundamentals", {})

    style_str = ", ".join(tags.get("style", [])).lower()
    region_str = ", ".join(tags.get("region", []))
    asset_class = ", ".join(tags.get("asset_class", [])).lower()
    sector = ", ".join(tags.get("sector_focus", [])).lower()
    factor_tilts = ", ".join(tags.get("factor_tilts", [])) or "none"

    desc = f"The {name} is a {style_str} {asset_class} index focused on the {region_str} region. "

    if "num_constituents" in fundamentals:
        desc += (
            f"It consists of approximately {fundamentals['num_constituents']} constituents "
        )
    if tags.get("weighting_method"):
        desc += f"and uses a {tags['weighting_method'].lower()} weighting methodology. "

    if sector:
        desc += f"This index targets the {sector} sector and "
    else:
        desc += "It "
    if "rebalance_frequency" in fundamentals and "rebalance_dates" in fundamentals:
        dates = ", ".join(fundamentals["rebalance_dates"])
        desc += (
            f"rebalances {fundamentals['rebalance_frequency'].lower()} in {dates}. "
        )

    if "dividend_yield" in fundamentals:
        desc += (
            f"It has a dividend yield of around {fundamentals['dividend_yield']}% "
        )
    if "pe_ratio" in fundamentals:
        desc += f"and a price-to-earnings (P/E) ratio of {fundamentals['pe_ratio']}. "

    if factor_tilts != "none":
        desc += f"It incorporates factor tilts such as {factor_tilts}. "
    else:
        desc += "It does not apply any explicit factor tilts. "

    desc += f"ESG focus: {'Yes' if tags.get('esg') else 'No'}. "
    if entry.get("account_minimum") is not None:
        desc += f"Minimum account size required is ${entry['account_minimum']:,}."

    return desc
