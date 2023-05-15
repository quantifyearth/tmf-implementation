%%%
Title = "Permanence Calculation"
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

This document defines the core methods, assumptions, inputs and outputs for calculating the permanence based on inputs from
additionality and leakage [@perm].

{mainmatter}

# Overview

Calculating permanence allows impermanent carbon reduction to be directly compared with permanent drawdown.

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT",
"SHOULD", "SHOULD NOT", "RECOMMENDED", "MAY", and "OPTIONAL" in this
document are to be interpreted as described in [@RFC2119].

# Inputs

The set of inputs that will be needed to calculate the desired outputs. Precomputed inputs are those likely coming from an external source (e.g. European Space Agency). Project-specific inputs are provided by those running avoided tropical moist forest deforestation projects.

## Precomputed Inputs

The Social Cost of Carbon (SCC) is needed as the Social Value of Offsets (SVO) [@scc]. SVO is a well-defined fraction of the Social Cost of Carbon which depends on the offset's expected lifetime, risk of non-additionality and risk of failure.

## Project-specific Inputs

To calculate permanence the methodologies for calculating additionality (TODO: cite additionality RFC) and leakage (TODO: cite leakage RFC) **MUST** have been followed for the project area.

# Outputs

TODO

# Method

The calculations for permanence must take place in the window of time between the end of the evaluation period `t_end` and the closest time period before `end` where estimates of additionality and leakage are available `t_prev`.

## Net Sequestration {#net-seq}

Whether a project sequesters or releases carbon in a given year is calculated by taking the difference between additionality and leakage for that given period. If \\(t_0, t_1, ..., t_i\\) ranges over the years a project has
been running, where \\(t_i\\) is *end of the evaluation period* we can calculate a given year's net sequestration as:

\\(C(t\_i) = (Add(t\_i) - Leak(t\_i)) - (Add(t\_{i - 1}) - Leak(t\_{i - 1}))\\)

Where \\(Add(t)\\) and \\(Leak(t)\\) are the project's additionality and leakage for year \\(t\\) respectively. A negative value denotes a release of carbon, whereas a positive value denotes sequestation.


<!-- CITATATIONS -->
<reference anchor='scc' target='https://doi.org/10.21203/rs.3.rs-1515075/v1'>
    <front>
        <title>The Social Value of Offsets</title>
        <author>
            <firstname>B</firstname>
            <lastname>Groom</lastname>
        </author>
        <author>
            <firstname>F</firstname>
            <lastname>Venmans</lastname>
        </author>
    </front>
</reference>
<reference anchor='perm' target='https://www.cambridge.org/engage/coe/article-details/63d404e96bc5cabaa41d2628'>
    <front>
        <title>The value of impermanent carbon credits</title>
        <author>
            <firstname>A</firstname>
            <lastname>Balmford</lastname>
        </author>
        <author>
            <firstname>S</firstname>
            <lastname>Keshav</lastname>
        </author>
        <author>
            <firstname>F</firstname>
            <lastname>Venmans</lastname>
        </author>
        <author>
            <firstname>D</firstname>
            <lastname>Coomes</lastname>
        </author>
        <author>
            <firstname>B</firstname>
            <lastname>Groom</lastname>
        </author>
        <author>
            <firstname>A</firstname>
            <lastname>Madhavapeddy</lastname>
        </author>
        <author>
            <firstname>T</firstname>
            <lastname>Swinfield</lastname>
        </author>
    </front>
</reference>

{backmatter}