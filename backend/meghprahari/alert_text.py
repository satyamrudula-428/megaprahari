"""Alert wording per language (task J4). English reproduces the original alert text exactly; Hindi (hi-IN) is a DRAFT
translation that must be reviewed by a native speaker / the issuing authority before any public use.
render() returns the CAP <info> texts (event, headline, description, instruction) for one language."""

LANG_CODES = {"en": "en-IN", "hi": "hi-IN"}
LEVEL_NAMES = {"en": {1: "WATCH", 2: "WARNING", 3: "ACT_NOW"},
               "hi": {1: "सतर्क रहें", 2: "चेतावनी", 3: "तुरंत कार्रवाई करें"}}
EVENT_NAMES = {"en": {"ts": "Severe thunderstorm", "cb": "Cloudburst-type extreme rainfall", "ff": "Flash flood"},
               "hi": {"ts": "गंभीर आंधी-तूफ़ान", "cb": "बादल फटने जैसी अत्यधिक वर्षा", "ff": "अचानक बाढ़"}}
COMPASS = {"en": ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"],
           "hi": ["उत्तर", "उत्तर-पूर्व", "पूर्व", "दक्षिण-पूर्व", "दक्षिण", "दक्षिण-पश्चिम", "पश्चिम", "उत्तर-पश्चिम"]}


def direction(lang, bearing_deg):
    return COMPASS[lang][int((bearing_deg + 22.5) % 360 // 45)]


def render(lang, *, level, hazard, village, prob, lead_start_min, lead_end_min, drivers, margin, walk,
           dissemination_min, refuge_dist_m, refuge_bearing_deg):
    """{'language', 'event', 'headline', 'description', 'instruction'} for one language. margin/walk in minutes (None =
    unknown), refuge distance in metres, bearing in degrees."""
    if lang not in LANG_CODES:
        raise ValueError(f"no alert templates for language '{lang}'; available: {sorted(LANG_CODES)}")
    event = EVENT_NAMES[lang][hazard]
    has_refuge = walk is not None and refuge_bearing_deg is not None
    if lang == "en":
        desc = (f"Model probability {prob:.0%} of {event.lower()} affecting the catchment upstream of {village} "
                f"beginning {lead_start_min}-{lead_end_min} min from now. Main drivers: {'; '.join(drivers) or 'n/a'}.")
        if margin is None:
            desc += " Time needed to reach higher ground is unknown (no refuge mapped for this village)."
        elif margin < 0:
            desc += (f" The forecast window may be shorter than the time needed to reach higher ground "
                     f"(short by about {-margin:.0f} min).")
        else:
            desc += (f" Estimated time to spare: about {margin:.0f} min (forecast lead minus {dissemination_min:g} min "
                     f"to deliver this message and about {walk:.0f} min on foot to higher ground).")
        instr = "Follow instructions from local authorities."
        if has_refuge:
            instr = (f"If told to act, move toward higher ground to the {direction(lang, refuge_bearing_deg)} "
                     f"(about {refuge_dist_m:.0f} m; roughly {walk:.0f} min on foot). " + instr)
        headline = f"{LEVEL_NAMES[lang][level]}: possible {event.lower()} near {village}"
    else:
        desc = (f"मॉडल के अनुसार {village} के ऊपरी जलग्रहण क्षेत्र में अब से {lead_start_min}-{lead_end_min} मिनट के "
                f"भीतर {event} की संभावना {prob:.0%} है।")
        if margin is None:
            desc += " ऊँची जगह तक पहुँचने में लगने वाला समय ज्ञात नहीं है (इस गाँव के लिए सुरक्षित स्थान चिह्नित नहीं है)।"
        elif margin < 0:
            desc += f" ऊँची जगह तक पहुँचने के लिए समय कम पड़ सकता है (लगभग {-margin:.0f} मिनट कम)।"
        else:
            desc += (f" अनुमानित बचा हुआ समय: लगभग {margin:.0f} मिनट (संदेश पहुँचने में {dissemination_min:g} मिनट और "
                     f"ऊँची जगह तक पैदल लगभग {walk:.0f} मिनट)।")
        instr = "स्थानीय प्रशासन के निर्देशों का पालन करें।"
        if has_refuge:
            instr = (f"निर्देश मिलने पर {direction(lang, refuge_bearing_deg)} दिशा में ऊँची जगह की ओर जाएँ "
                     f"(लगभग {refuge_dist_m:.0f} मीटर; पैदल लगभग {walk:.0f} मिनट)। " + instr)
        headline = f"{LEVEL_NAMES[lang][level]}: {village} के पास {event} की आशंका"
    return dict(language=LANG_CODES[lang], event=event, headline=headline, description=desc, instruction=instr)
