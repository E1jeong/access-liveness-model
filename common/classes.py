CLASS_NAMES = [
    "live",
    "print",
    "picture",
    "mask",
    "display",
    "pmask",
    "curved_print",
    "curved_mask",
    "curved_picture",
    "curved_pmask",
    "dental_white",
    "dental_black",
]
CLASS_MAPPING = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Access-control policy: a person wearing either approved dental mask remains
# a bona-fide presentation. The multiclass output order above remains the
# deployment contract; these groups drive PAD metrics and auxiliary targets.
BONA_FIDE_CLASS_NAMES = ("live", "dental_white", "dental_black")
BONA_FIDE_CLASS_INDICES = tuple(CLASS_MAPPING[name] for name in BONA_FIDE_CLASS_NAMES)
ATTACK_CLASS_INDICES = tuple(
    index for index in range(len(CLASS_NAMES)) if index not in BONA_FIDE_CLASS_INDICES
)
