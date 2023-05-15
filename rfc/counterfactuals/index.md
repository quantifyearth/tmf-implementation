%%%
Title = "Counterfactual Generation"
area = "Project Level"
workgroup = "Avoided tropical moist forest deforestation"

[seriesInfo]
name = "Internet-Draft"
value = "draft-atmfd-00"
stream = "IETF"
status = "informational"

date = 2018-08-21T00:00:00Z

[[author]]
initials="C."
surname="Bactrain"
fullname="C. Bactrian"
%%%


.# Abstract

This document defines the core methods, assumptions, inputs and outputs for generating candidate counterfactual pixels
for use in assessing the additionality and leakage of avoided tropical moist forest deforestation projects.

{mainmatter}

# Overview

Counterfactual points are needed to calculate additionality and leakage of avoided tropical moist forest deforestation projects. Counterfactuals form a control group with which we can compare a treatment group (the project) to assess the impact of the interventions. Some more information about the use of counterfactuals written by people that understand the space better.

The project area is modelled as a population of 30m by 30m pixels, where each pixel has an Annual Forest Change (AFC) land use class (LUC), such as 'Undisturbed' or 'Deforested' [@afc]. We use GEDI shots to estimate the carbon density of each land use class (Section 7.1). We then match each project pixel to 100 counterfactual pixels, and compute the total biomass in the project and the mean total biomass in the counterfactual from their corresponding land use classes.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in [@RFC2119].

# Inputs

The set of inputs that will be needed to calculate the desired outputs. Precomputed inputs are those likely coming from an external source (e.g. European Space Agency). Project-specific inputs are provided by those running avoided tropical moist forest deforestation projects.

## Precomputed Inputs

The GEDI L4A Footprint level aboveground biomass density data is **REQUIRED** for these calculations [@gedi4a]. It **SHOULD** be the most up to date version currently available.

The AFC LUC data is **REQUIRED** at a 30m by 30m resolution [@afc]. For clarity, this data **MUST** include the following six land classes:

 1. Undisturbed
 2. Degraded
 3. Deforested
 4. Regrowth
 5. Water
 6. Other

## Project-specific Inputs

The method requires a set of pixels for the project at a resolution of 30m by 30m. Each pixel **MUST** be accompanied by the AFC Land Use Class (LUC) [@afc]. Where a pixel may lie on the boundary of a project, it **SHOULD** be included in the set of project pixels.

# Outputs

## Project-specific values for Land Use Cover and Biomass

TODO: explanation why we might want this, how it is used etc.

## Counterfactual Points for a Project

TODO: explanation why we might want this, how it is used etc.

# Method

## Coarse, Proportional Land Cover Patches

Using the AFC LUC data [@afc], 1200m by 1200m patches can be produced containing the proportional land cover. Each patch will therefore contain 1600 of the 30m by 30m AFC pixels. For a particular land class `lc`, this might look like the following.

```
acc = 0
for i = 0 to 40 do
    for j = 0 to 40 do
        acc += (if afc[i][j] = lc then 1 else 0)
    done;
done;
acc = acc / 1600
```

`acc` now contains the proportional value for `lc` for a particular 1200m by 1200m patch.

## Calculating the buffer

A project is defined by the set of pixels. A set of polygons may be inferred combining collections of adjacent pixels into distinct polygons. Where project polygons are already available, these **SHOULD** be used instead. There are three scenarios for project polygons.

### Polygons with interior rings

Projects may have inner polygons to represent areas that are contained by the project, but not a part of the project itself (forming a doughnut shape). For the sake of buffer calculatations these should be ignored and the polygon(s) treated as completed polygons. The method from this point is the same as in (#multiple-polygons) or (#single-polygon).

### Multiple polygons {#multiple-polygons}

When there are multiple, non-overlapping polygons, these **MUST** be combined into one large project polygon for the sake of generating the buffer area. This can be done by taking the convex hull of all of the points that make up the exterior ring of the polygons. Once this is done, the method is the same as in (#single-polygon).

### A single polygon {#single-polygon}

Assuming the case is now that we have a single, concave polygon representing the project the buffer can be calculated using the standard method of taking Mikowski sum of the polygon with a circle and then calculating the convex hull of the shape. (TODO: be clear about how much approximation should occur and how to merge shapes).

## Values for Land Use Classes and Above Ground Biomass

Estimates are made for the above ground biomass (AGB) for the different LUCs found in the buffer zone. We use the GEDI L4A dataset [@gedi4a] as basis for these calculations (see the explanation for this assumption in (#a-gedi-1)).

Using the GEDI L4A data [@gedi4a] along with the buffer from (#single-polygon), a set of GEDI shots can be calculated that fall inside the buffer (but not inside the project area). This set is denoted by `Gedi_shots`.

A new set of shots, `Gedi_filtered`, is then constructed by filtering `Gedi_shots` using the following critera:

 1. The `degrade_flag` of the shot is equal to `0`.
 2. The `beam_type` of the shot is equal to `'full'`.
 3. The `l4_quality_flag` is equal to `1`.
 4. The `flag` of the shot is NOT equal to `'leaf-off state'`.

This assumption behind this filtering algorithm are explained in (#a-gedi-2).

For all shots `s` in `Gedi_filtered` a mapping from `s` to its AFC LUC [@afc] should be constructed. We further filter the `Gedi_filtered` set by removing those shots `s` whose 8 adjacent neighbours differ in LUC from that of `s`, let this new filtered set be called `Gedi_final`.

```
+---+---+---+
|nb1|nb2|nb3|
+---+---+---+
|nb4| s |nb5|   for all n in { nb1, nb2, nb3, nb4, nb5, nb6, nb7, nb8 }.
+---+---+---+       s.land_use_class = n.land_use_class
|nb6|nb7|nb8|
+---+---+---+
```

When computing `Gedi_final` from `Gedi_filtered`, neighbouring pixels **MUST** also be in `Gedi_filtered`. When this is not the case *we do something that I don't know or maybe the assumption just made was completely incorrect*.

`Gedi_final` now contains a set of GEDI shots (pixels). For each shot `s` in `Gedi_final`, we should have a LUC value (`s.land_use_class`) and an above ground biomass density (AGB) value (`s.agbd`). The latter coming from GEDI.

For each of the possible LUCs we calculate the median AGB density value for the shots in `Gedi_final`. Values for below ground biomass (BGB) and deadwood biomass are assumed to be 20% and 11% of AGB respectively [@cairns] [@ipcc2003].

## Generating Counterfactuals

TODO

# Assumptions

The following sectadions outline a set of assumptions that this methodology makes and the justifications for those assumptions.

## GEDI L4A accurately measures AGB {#a-gedi-1}

This is the state of the art and can be replaced by better estimates as they become available.

TODO: a better justification/cite

## GEDI L4A shots are correctly filtered {#a-gedi-2}

This is the best publicly-known setting right now.

TODO: a better justification/cite

<!-- CITATATIONS -->
<reference anchor='afc' target='https://www.science.org/doi/10.1126/sciadv.abe1603'>
    <front>
        <title>Long-term (1990–2019) monitoring of forest cover changes in the humid tropics</title>
        <author>
            <firstname>Vancutsem</firstname>
            <lastname>Christelle</lastname>
        </author>
        <author>
            <firstname>Achard</firstname>
            <lastname>Frédéric</lastname>
        </author>
        <author>
            <firstname>Pekel</firstname>
            <lastname>J-F</lastname>
        </author>
        <author>
            <firstname>Vieilledent</firstname>
            <lastname>Ghislain</lastname>
        </author>
        <author>
            <firstname>Carboni</firstname>
            <lastname>S</lastname>
        </author>
        <author>
            <firstname>Simonetti</firstname>
            <lastname>Dario</lastname>
        </author>
        <author>
            <firstname>Gallego</firstname>
            <lastname>Javier</lastname>
        </author>
        <author>
            <firstname>Aragao</firstname>
            <lastname>Luiz EOC</lastname>
        </author>
        <author>
            <firstname>Nasi</firstname>
            <lastname>Robert</lastname>
        </author>
    </front>
</reference>

<!-- Check this ref! -->
<reference anchor='gedi4a' target='https://doi.org/10.3334/ORNLDAAC/2056'>
  <front>
    <title>GEDI L4A Footprint Level Aboveground Biomass Density, Version 2.1</title>
    <author>
    <firstname>Dubayah</firstname>
    <lastname>R.O.</lastname>
    </author>
    <author>
    <firstname>Armston</firstname>
    <lastname>J.</lastname>
    </author>
    <author>
    <firstname>Kellner</firstname>
    <lastname>J.R.</lastname>
    </author>
    <author>
    <firstname>Duncanson</firstname>
    <lastname>L.</lastname>
    </author>
    <author>
    <firstname>Healey</firstname>
    <lastname>S.P.</lastname>
    </author>
    <author>
    <firstname>Patterson</firstname>
    <lastname>P.L.</lastname>
    </author>
    <author>
    <firstname>Hancock</firstname>
    <lastname>S.</lastname>
    </author>
    <author>
    <firstname>Tang</firstname>
    <lastname>H.</lastname>
    </author>
    <author>
    <firstname>Bruening</firstname>
    <lastname>J.</lastname>
    </author>
    <author>
    <firstname>Hofton</firstname>
    <lastname>M.A.</lastname>
    </author>
    <author>
    <firstname>Blair</firstname>
    <lastname>J.B.</lastname>
    </author>
    <author>
    <firstname>Luthcke</firstname>
    <lastname>S.B.</lastname>
    </author>
    <year>2022</year>
  </front>
</reference>

<reference anchor='cairns' target='https://link.springer.com/article/10.1007/s004420050201'>
  <front>
    <title>Root biomass allocation in the world's upland forests</title>
    <author>
    <firstname>Cairns</firstname>
    <lastname>Michael A</lastname>
    </author>
    <author>
    <firstname>Brown</firstname>
    <lastname>Sandra</lastname>
    </author>
    <author>
    <firstname>Helmer</firstname>
    <lastname>Eileen H</lastname>
    </author>
    <author>
    <firstname>Baumgardner</firstname>
    <lastname>Greg A</lastname>
    </author>
  </front>
</reference>

<reference anchor='ipcc2003' target='https://www.ipcc.ch/publication/good-practice-guidance-for-land-use-land-use-change-and-forestry/'>
    <front>
        <title>Good Practice Guidance for Land Use, Land-Use Change and Forestry</title>
        <author>
        <firstname>IPCC</firstname>
        </author>
    </front>
</reference>

{backmatter}