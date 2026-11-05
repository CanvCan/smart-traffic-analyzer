"""
traffic_analyzer/domain/theme.py

Shared color palette and typography tokens for desktop and visualization UI.
"""

from dataclasses import dataclass

SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 20
SPACE_6 = 24

CORNER_RADIUS_SM = 6
CORNER_RADIUS_MD = 10
BORDER_WIDTH = 1


@dataclass(frozen=True)
class ThemePalette:
    bg_root: str = "#0c0e1a"
    bg_panel: str = "#10132a"
    bg_elevated: str = "#161930"
    bg_input: str = "#10132a"
    bg_hover: str = "#1d2040"
    bg_selected: str = "#111535"

    border_subtle: str = "#252a50"
    border_strong: str = "#3a3f6a"
    border_focus: str = "#6070e8"

    text_primary: str = "#d6daf8"
    text_secondary: str = "#7880b8"
    text_muted: str = "#4a5080"

    accent_primary: str = "#6070e8"
    accent_hover: str = "#5060d8"
    accent_muted: str = "#111535"

    status_ok: str = "#3dd68c"
    status_ok_bg: str = "#0a2218"
    status_warn: str = "#e8a020"
    status_warn_bg: str = "#251a05"
    status_danger: str = "#e05575"
    status_danger_bg: str = "#2a1018"

    canvas_bg: str = "#0c0e1a"
    canvas_grid: str = "#1e2244"
    vertex_fill: str = "#d6daf8"
    vertex_selected: str = "#e8a020"
    vertex_hover: str = "#4db8cc"
    snap_guide: str = "#4db8cc"

    @property
    def success(self) -> str:
        return self.status_ok

    @property
    def danger(self) -> str:
        return self.status_danger


@dataclass(frozen=True)
class ThemeTypography:
    family_sans: str = "Plus Jakarta Sans"
    family_mono: str = "Consolas"
    size_xs: int = 10
    size_sm: int = 11
    size_md: int = 12
    size_lg: int = 14
    size_xl: int = 17


PALETTE = ThemePalette()
TYPOGRAPHY = ThemeTypography()

BG = PALETTE.bg_root
SURFACE = PALETTE.bg_panel
CARD = PALETTE.bg_elevated
CARD_HI = PALETTE.bg_hover
BORDER = PALETTE.border_subtle
SEL_BG = PALETTE.bg_selected
SEL_BOR = PALETTE.border_focus

FG = PALETTE.text_primary
FG_MED = PALETTE.text_secondary
FG_DIM = PALETTE.text_muted

ACCENT = PALETTE.accent_primary
ACCENT_H = PALETTE.accent_hover
GREEN = PALETTE.status_ok
GREEN_D = PALETTE.status_ok_bg
YELLOW = PALETTE.status_warn
YELLOW_D = PALETTE.status_warn_bg
RED = PALETTE.status_danger
RED_D = PALETTE.status_danger_bg

# Semantic aliases for visualization components
BG_PRIMARY = BG
BG_SECONDARY = SURFACE
BG_TERTIARY = CARD
BG_ELEVATED = CARD_HI
BORDER_SUBTLE = BORDER
ACCENT_PRIMARY = ACCENT
ACCENT_HOVER = ACCENT_H
ACCENT_SUCCESS = GREEN
ACCENT_SUCCESS_HOVER = "#2eb872"
ACCENT_WARNING = YELLOW
ACCENT_DANGER = RED
ACCENT_DANGER_HOVER = "#c44160"
TEXT_PRIMARY = FG
TEXT_SECONDARY = FG_MED
TEXT_MUTED = FG_DIM

FONT_FAMILY = TYPOGRAPHY.family_sans
F_TITLE = (FONT_FAMILY, 16, "bold")
F_HEAD = (FONT_FAMILY, 12, "bold")
F_BODY = (FONT_FAMILY, 11)
F_SMALL = (FONT_FAMILY, 10)
F_HELPER = (FONT_FAMILY, 9)
F_MONO = (TYPOGRAPHY.family_mono, 10)

LANE_HEX_PALETTE = (
    "#6070e8",
    "#3dd68c",
    "#e8a020",
    "#e05575",
    "#4db8cc",
    "#8B5CF6",
    "#84CC16",
    "#F97316",
)
