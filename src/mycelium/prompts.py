"""
E06 generation prompts -- 5 paraphrases per cell, round-robined across 100 seeds
(seed index i uses paraphrase i % 5), exactly matching E05's established mechanism
(generate_rural_road_e05.py) rather than inventing a new one (R4). Same seed budget,
zero extra images -- addresses the prompt-wording risk MetaLogic documented (FLUX.dev
misaligns 59% of the time under pure grammatical rephrasing, arXiv:2510.00796, cited in
RECOMMENDATIONS_E04 SS4) without adding cost, per the user's explicit choice (2026-08-16)
to split the 50->100 seed increase across paraphrases rather than one fixed string.

Rural paraphrases (both countries): reused VERBATIM from generate_rural_road_e05.py's
"explicit" mode list -- paraphrase #1 in each is a word-for-word match with
model_comparison_test/config_snapshot.json's single rural prompt, confirming the exact
same lineage/content, not independently re-derived.

Urban paraphrases: model_comparison_test's single urban string is paraphrase #1; 4 more
written to match its exact content (every detail preserved: paved street + lane markings,
multi-story buildings with shopfronts + illegible signage, the specific vehicles named,
overhead wires/streetlights, roadside trees, sidewalks + pedestrians, lighting/sky, "no
legible text or landmarks", "photographed from street level") in the same
grammatical-variation style as the rural set -- same facts, different sentence structure,
nothing added or removed.

    from prompts import PROMPTS, prompt_for_seed
    prompt = prompt_for_seed("india_urban_commercial_street", seed_index=37)
"""

N_SEEDS = 250  # revised 2026-08-18, was 100 -- user's call: match the real reference set's
# 250/cell so all three sets (real/oob/finetuned) are the same size, 1000 images each

PROMPTS = {
    "india_generic_rural_road": [
        "A rural road in India: a quiet two-lane paved road with faded white lane "
        "markings, curving gently and receding into the distance, dry semi-arid roadside "
        "vegetation with sparse dusty shrubs and reddish-brown earth, occasional palm "
        "trees, harsh midday sun under a pale hazy sky, utility poles with power lines "
        "running alongside, at most one vehicle in view, no legible signs or landmarks, "
        "photographed from inside a moving car",
        "In India, a quiet two-lane paved rural road curves gently into the distance; "
        "faded white lane markings, dry semi-arid vegetation with sparse dusty shrubs, "
        "reddish-brown earth, and occasional palm trees line the roadside under a harsh "
        "midday sun and pale hazy sky. Utility poles carry power lines alongside. At most "
        "one vehicle is visible, with no legible signs or landmarks. Shot from inside a "
        "moving car.",
        "Photograph of an Indian rural road: two paved lanes with faded white markings "
        "bend gently away into the distance. The roadside is dry and semi-arid, dotted "
        "with dusty shrubs, reddish-brown soil, and the occasional palm tree, under a "
        "hazy, sun-bleached sky. Power lines run along utility poles beside the road. No "
        "more than one vehicle appears, no legible text or landmarks, taken from a moving "
        "car's interior.",
        "A quiet paved rural road in India curving into the distance, two lanes with "
        "faded white markings, bordered by sparse dusty shrubs, reddish earth, and the "
        "occasional palm tree under harsh midday sun and a pale hazy sky. Power lines "
        "strung along utility poles run beside the road. At most one vehicle is visible; "
        "no landmarks or readable signs appear. Photographed from a moving car.",
        "From inside a moving car in India: a two-lane paved rural road, faded white "
        "lane markings, curving off into the distance, dry semi-arid roadside scrub, "
        "reddish-brown earth, an occasional palm tree, harsh midday sunlight, a pale "
        "hazy sky, and utility poles carrying power lines alongside. No more than one "
        "vehicle in frame, no legible signage or landmarks.",
    ],
    "uk_generic_rural_road": [
        "A rural road in the United Kingdom: a quiet two-lane paved road with faded "
        "white lane markings, curving gently and receding into the distance, lush green "
        "grassy verges and low hedgerows, damp asphalt under an overcast grey sky, "
        "temperate deciduous trees, utility poles with power lines running alongside, at "
        "most one vehicle in view, no legible signs or landmarks, photographed from "
        "inside a moving car",
        "In the United Kingdom, a quiet two-lane paved rural road curves gently into the "
        "distance; faded white lane markings, lush green grassy verges, low hedgerows, "
        "and temperate deciduous trees line the roadside, with damp asphalt under an "
        "overcast grey sky. Utility poles carry power lines alongside. At most one "
        "vehicle is visible, with no legible signs or landmarks. Shot from inside a "
        "moving car.",
        "Photograph of a rural road in the United Kingdom: two paved lanes with faded "
        "white markings bend gently away into the distance. The roadside is lush and "
        "green, with low hedgerows and grassy verges, and temperate deciduous trees, "
        "under a damp, overcast grey sky. Power lines run along utility poles beside the "
        "road. No more than one vehicle appears, no legible text or landmarks, taken "
        "from a moving car's interior.",
        "A quiet paved rural road in the United Kingdom curving into the distance, two "
        "lanes with faded white markings, bordered by lush grassy verges, low hedgerows, "
        "and temperate deciduous trees under a damp, overcast grey sky. Power lines "
        "strung along utility poles run beside the road. At most one vehicle is visible; "
        "no landmarks or readable signs appear. Photographed from a moving car.",
        "From inside a moving car in the United Kingdom: a two-lane paved rural road, "
        "faded white lane markings, curving off into the distance, lush green grassy "
        "verges, low hedgerows, temperate deciduous trees, damp asphalt, an overcast "
        "grey sky, and utility poles carrying power lines alongside. No more than one "
        "vehicle in frame, no legible signage or landmarks.",
    ],
    "india_urban_commercial_street": [
        "An urban commercial street in India: a paved street with visible lane markings, "
        "lined on both sides by multi-story buildings with shopfronts and densely packed "
        "signboards and hoardings (illegible), auto-rickshaws, motorcycles and scooters "
        "moving among cars and the occasional bus, tangled overhead wires strung between "
        "utility poles, roadside trees including the occasional palm tree, uneven "
        "concrete sidewalks with a few pedestrians, harsh midday sun under a pale hazy "
        "sky, no legible text or identifiable landmarks, photographed from street level",
        "In India, an urban commercial street runs between multi-story buildings crowded "
        "with shopfronts and densely packed signboards and hoardings (illegible); the "
        "paved street carries visible lane markings as auto-rickshaws, motorcycles, and "
        "scooters weave among cars and the occasional bus. Tangled overhead wires "
        "stretch between utility poles, roadside trees including the occasional palm "
        "tree line the street, and uneven concrete sidewalks carry a few pedestrians "
        "under a harsh midday sun and pale hazy sky. No legible text or identifiable "
        "landmarks. Shot from street level.",
        "Photograph of an Indian urban commercial street: a paved road with visible lane "
        "markings runs between multi-story buildings whose shopfronts are packed with "
        "illegible signboards and hoardings. Auto-rickshaws, motorcycles, and scooters "
        "move among cars and the occasional bus, beneath tangled overhead wires strung "
        "between utility poles. Roadside trees, including the occasional palm, and "
        "uneven concrete sidewalks with a few pedestrians complete the scene under a "
        "harsh, hazy midday sun. No legible text or landmarks, taken from street level.",
        "A busy commercial street in urban India, paved with visible lane markings and "
        "flanked by multi-story buildings dense with shopfronts and illegible signboards "
        "and hoardings. Auto-rickshaws, motorcycles, and scooters thread between cars and "
        "the occasional bus, under tangled overhead wires on utility poles. Roadside "
        "trees, an occasional palm among them, shade uneven concrete sidewalks where a "
        "few pedestrians walk, all under a hazy midday sun. No legible text or "
        "identifiable landmarks. Photographed at street level.",
        "At street level on an urban commercial street in India: a paved road with "
        "visible lane markings, multi-story buildings on both sides packed with "
        "shopfronts and illegible signboards and hoardings, auto-rickshaws, motorcycles, "
        "and scooters moving among cars and the occasional bus, tangled overhead wires "
        "between utility poles, roadside trees including an occasional palm, uneven "
        "concrete sidewalks with a few pedestrians, and a harsh midday sun under a pale "
        "hazy sky. No legible text or landmarks.",
    ],
    "uk_urban_commercial_street": [
        "An urban commercial street in the United Kingdom: an asphalt street marked with "
        "white lane lines, lined on both sides by red-brick multi-story buildings with "
        "shopfronts and storefront signage (illegible), a red double-decker bus and a "
        "few parked cars and vans along the curb, tidy paved sidewalks with black iron "
        "streetlights and a few pedestrians, damp asphalt under an overcast grey sky, "
        "roadside trees, no legible signs or identifiable landmarks, photographed from "
        "street level",
        "In the United Kingdom, an urban commercial street is lined on both sides by "
        "red-brick multi-story buildings with shopfronts and illegible storefront "
        "signage; the asphalt street is marked with white lane lines. A red "
        "double-decker bus and a few parked cars and vans sit along the curb, beside "
        "tidy paved sidewalks with black iron streetlights and a few pedestrians. The "
        "asphalt is damp under an overcast grey sky, with roadside trees nearby. No "
        "legible signs or identifiable landmarks. Shot from street level.",
        "Photograph of a UK urban commercial street: an asphalt road marked with white "
        "lane lines runs between red-brick multi-story buildings whose shopfronts carry "
        "illegible signage. A red double-decker bus and a few parked cars and vans line "
        "the curb, and tidy paved sidewalks with black iron streetlights carry a few "
        "pedestrians. Roadside trees stand nearby under a damp, overcast grey sky. No "
        "legible signs or landmarks, taken from street level.",
        "A commercial street in urban Britain, asphalt marked with white lane lines and "
        "flanked by red-brick multi-story buildings dense with shopfronts and illegible "
        "storefront signage. A red double-decker bus and a handful of parked cars and "
        "vans sit at the curb, near tidy paved sidewalks lit by black iron streetlights "
        "where a few pedestrians walk. Roadside trees stand under a damp, overcast grey "
        "sky. No legible signs or identifiable landmarks. Photographed at street level.",
        "At street level on an urban commercial street in the United Kingdom: an asphalt "
        "road marked with white lane lines, red-brick multi-story buildings on both "
        "sides with shopfronts and illegible storefront signage, a red double-decker bus "
        "and a few parked cars and vans along the curb, tidy paved sidewalks with black "
        "iron streetlights and a few pedestrians, roadside trees, and damp asphalt under "
        "an overcast grey sky. No legible signs or landmarks.",
    ],
}

for _cell, _plist in PROMPTS.items():
    assert len(_plist) == 5, f"{_cell}: expected 5 paraphrases, got {len(_plist)}"


def prompt_for_seed(cell: str, seed_index: int) -> tuple[int, str]:
    """Returns (paraphrase_index, prompt_text). Round-robin: seed_index i always maps to
    the same paraphrase i % 5, matching generate_rural_road_e05.py's mechanism exactly --
    so guidance/condition comparisons at the same seed always use the same wording too."""
    idx = seed_index % 5
    return idx, PROMPTS[cell][idx]
