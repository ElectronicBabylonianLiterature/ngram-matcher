from typing import Mapping
from enum import Enum, unique


def get_by_attribute_value(cls, attribute, value):
    try:
        return next(enum for enum in cls if getattr(enum, attribute) == value)
    except StopIteration:
        raise ValueError(f"Unknown {cls.__name__}.{attribute}: {value}")


class NamedEnum(Enum):
    def __init__(self, long_name, abbreviation, sort_key=-1):
        self.long_name = long_name
        self.abbreviation = abbreviation
        self.sort_key = sort_key

    @classmethod
    def from_abbreviation(cls, abbreviation):
        return get_by_attribute_value(cls, "abbreviation", abbreviation)

    @classmethod
    def from_name(cls, name):
        return get_by_attribute_value(cls, "long_name", name)


class NamedEnumWithParent(NamedEnum):
    def __init__(self, long_name, abbreviation, parent, sort_key=-1):
        super().__init__(long_name, abbreviation, sort_key)
        self.parent = parent


class ManuscriptType(NamedEnum):
    LIBRARY = ("Library", "")
    SCHOOL = ("School", "Sch")
    VARIA = ("Varia", "Var")
    COMMENTARY = ("Commentary", "Com")
    QUOTATION = ("Quotation", "Quo")
    EXCERPT = ("Excerpt", "Ex")
    PARALLEL = ("Parallel", "Par")
    NONE = ("None", "")


class Period(NamedEnumWithParent):
    NONE = ("None", "", None, 0)
    UNCERTAIN = ("Uncertain", "Unc", None, 1)
    URUK_IV = ("Uruk IV", "Uruk4", None, 2)
    URUK_III_JEMDET_NASR = ("Uruk III-Jemdet Nasr", "JN", None, 3)
    ED_I_II = ("ED I-II", "ED1_2", None, 4)
    FARA = ("Fara", "Fara", None, 5)
    PRESARGONIC = ("Presargonic", "PSarg", None, 6)
    SARGONIC = ("Sargonic", "Sarg", None, 7)
    UR_III = ("Ur III", "Ur3", None, 8)
    OLD_ASSYRIAN = ("Old Assyrian", "OA", None, 9)
    OLD_BABYLONIAN = ("Old Babylonian", "OB", None, 10)
    MIDDLE_BABYLONIAN = ("Middle Babylonian", "MB", None, 11)
    MIDDLE_ASSYRIAN = ("Middle Assyrian", "MA", None, 12)
    HITTITE = ("Hittite", "Hit", None, 13)
    NEO_ASSYRIAN = ("Neo-Assyrian", "NA", None, 14)
    NEO_BABYLONIAN = ("Neo-Babylonian", "NB", None, 15)
    LATE_BABYLONIAN = ("Late Babylonian", "LB", None, 16)
    PERSIAN = ("Persian", "Per", "Late Babylonian", 17)
    HELLENISTIC = ("Hellenistic", "Hel", "Late Babylonian", 18)
    PARTHIAN = ("Parthian", "Par", "Late Babylonian", 19)
    PROTO_ELAMITE = ("Proto-Elamite", "PElam", None, 20)
    OLD_ELAMITE = ("Old Elamite", "OElam", None, 21)
    MIDDLE_ELAMITE = ("Middle Elamite", "MElam", None, 22)
    NEO_ELAMITE = ("Neo-Elamite", "NElam", None, 23)


class ProvenanceEnum(NamedEnum):
    def __init__(self, long_name, abbreviation, parent, cigs_key, sort_key=-1):
        super().__init__(long_name, abbreviation, sort_key)
        self.parent = parent
        self.cigs_key = cigs_key


class Provenance(ProvenanceEnum):
    STANDARD_TEXT = ("Standard Text", "Std", None, None)
    ASSYRIA = ("Assyria", "Assa", None, None)
    ASSUR = ("Aššur", "Ašš", "Assyria", "ASS")
    DUR_KATLIMMU = ("Dūr-Katlimmu", "Dka", "Assyria", "DKA")
    GUZANA = ("Guzana", "Guz", "Assyria", "HLF")
    HARRAN = ("Ḫarrān", "Har", "Assyria", "HAR")
    HUZIRINA = ("Ḫuzirina", "Huz", "Assyria", "HUZ")
    IMGUR_ENLIL = ("Imgur-Enlil", "Img", "Assyria", "BLW")
    KALHU = ("Kalḫu", "Kal", "Assyria", "NIM")
    KAR_TUKULTI_NINURTA = ("Kār-Tukulti-Ninurta", "Ktn", "Assyria", "KTN")
    KHORSABAD = ("Khorsabad", "Kho", "Assyria", "SAR")
    NINEVEH = ("Nineveh", "Nin", "Assyria", "NNV")
    SUBAT_ENLIL = ("Šubat-Enlil", "Šub", "Assyria", "SZE")
    TARBISU = ("Tarbiṣu", "Tar", "Assyria", "SKH")
    BABYLONIA = ("Babylonia", "Baba", None, None)
    ADAB = ("Adab", "Adb", "Babylonia", "ADB")
    BABYLON = ("Babylon", "Bab", "Babylonia", "BAB")
    BAD_TIBIRA = ("Bad-Tibira", "Btb", "Babylonia", "BTB")
    BORSIPPA = ("Borsippa", "Bor", "Babylonia", "BOR")
    CUTHA = ("Cutha", "Cut", "Babylonia", "GUD")
    DILBAT = ("Dilbat", "Dil", "Babylonia", "DLB")
    DUR_KURIGALZU = ("Dūr-Kurigalzu", "Dku", "Babylonia", "AQA")
    ERIDU = ("Eridu", "Eri", "Babylonia", "ERI")
    ESNUNNA = ("Ešnunna", "Ešn", "Babylonia", "ESH")
    GARSANA = ("Garšana", "Gar", "Babylonia", "GRS")
    GIRSU = ("Girsu", "Gir", "Babylonia", "GIR")
    HURSAGKALAMA = ("Ḫursagkalama", "Hur", "Babylonia", None)
    IRISAGRIG = ("Irisagrig", "Irs", "Babylonia", "IRS")
    ISIN = ("Isin", "Isn", "Babylonia", "ISN")
    KISURRA = ("Kisurra", "Ksr", "Babylonia", "KSR")
    KIS = ("Kiš", "Kiš", "Babylonia", "KSH")
    KUTALLA = ("Kutalla", "Kut", "Babylonia", "SFR")
    LAGAS = ("Lagaš", "Lag", "Babylonia", "LAG")
    LARAK = ("Larak", "Lrk", "Babylonia", "LRK")
    LARSA = ("Larsa", "Lar", "Babylonia", "LAR")
    MALGIUM = ("Malgium", "Mal", "Babylonia", "TYA")
    MARAD = ("Marad", "Mrd", "Babylonia", "MRD")
    MASKAN_SAPIR = ("Maškan-šāpir", "Maš", "Babylonia", "MSK")
    METURAN = ("Meturan", "Met", "Babylonia", "HDD")
    NEREBUN = ("Nērebtum", "Nēr", "Babylonia", "NRB")
    NIGIN = ("Nigin", "Nig", "Babylonia", "NGN")
    NIPPUR = ("Nippur", "Nip", "Babylonia", "NIP")
    PI_KASI = ("Pī-Kasî", "Pik", "Babylonia", "ANT")
    PUZRIS_DAGAN = ("Puzriš-Dagān", "Puz", "Babylonia", "DRE")
    SIPPAR = ("Sippar", "Sip", "Babylonia", "SAP")
    SIPPAR_AMNANUM = ("Sippar-Amnānum", "Sipam", "Babylonia", "SIP")
    SADUPPUM = ("Šaduppûm", "Šad", "Babylonia", "SDP")
    SAHRINU = ("Šaḫrīnu", "Šah", "Babylonia", None)
    SURUPPAK = ("Šuruppak", "Šur", "Babylonia", "SUR")
    TUTUB = ("Tutub", "Ttb", "Babylonia", "TTB")
    UMMA = ("Umma", "Umm", "Babylonia", "JOK")
    UR = ("Ur", "Ur", "Babylonia", "URI")
    URUK = ("Uruk", "Urk", "Babylonia", "URU")
    ZABALAM = ("Zabalam", "Zab", "Babylonia", "ZAB")
    PERIPHERY = ("Periphery", "", None, None)
    ALALAKS = ("Alalakh", "Ala", "Periphery", "ALA")
    TELL_EL_AMARNA = ("Tell el-Amarna", "Ama", "Periphery", "AKH")
    ANSAN = ("Anšan", "Anš", "Periphery", "ANS")
    DER = ("Dēr", "Der", "Periphery", "DER")
    DUR_UNTAS = ("Dūr-Untaš", "Dun", "Periphery", "COZ")
    EBLA = ("Ebla", "Ebl", "Periphery", "EBA")
    ELAM = ("Elam", "Elam", "Periphery", None)
    EMAR = ("Emar", "Emr", "Periphery", "EMR")
    HATTUSA = ("Ḫattuša", "Hat", "Periphery", "HAT")
    KANES = ("Kaneš", "Kan", "Periphery", "KNS")
    KARKEMIS = ("Karkemiš", "Kar", "Periphery", "KRK")
    MARI = ("Mari", "Mar", "Periphery", "MAR")
    MEGIDDO = ("Megiddo", "Meg", "Periphery", "MGD")
    PASIME = ("Pašime", "Paš", "Periphery", "PAS")
    PERSEPOLIS = ("Persepolis", "Per", "Periphery", "PRS")
    QATNA = ("Qaṭnā", "Qaṭ", "Periphery", "QTN")
    SUSA = ("Susa", "Sus", "Periphery", "SUS")
    TEPE_GOTVAND = ("Tepe Gotvand", "Tgo", "Periphery", "GTV")
    TERQA = ("Terqa", "Ter", "Periphery", "TRQ")
    TUTTUL = ("Tuttul", "Ttl", "Periphery", "TUT")
    UGARIT = ("Ugarit", "Uga", "Periphery", "UGA")
    UNCERTAIN = ("Uncertain", "Unc", None, None)


@unique
class Stage(Enum):
    UNCERTAIN = "Uncertain"
    URUK_IV = "Uruk IV"
    URUK_III_JEMDET_NASR = "Uruk III-Jemdet Nasr"
    ED_I_II = "ED I-II"
    FARA = "Fara"
    PRESARGONIC = "Presargonic"
    SARGONIC = "Sargonic"
    UR_III = "Ur III"
    OLD_ASSYRIAN = "Old Assyrian"
    OLD_BABYLONIAN = "Old Babylonian"
    MIDDLE_BABYLONIAN = "Middle Babylonian"
    MIDDLE_ASSYRIAN = "Middle Assyrian"
    HITTITE = "Hittite"
    NEO_ASSYRIAN = "Neo-Assyrian"
    NEO_BABYLONIAN = "Neo-Babylonian"
    LATE_BABYLONIAN = "Late Babylonian"
    PERSIAN = "Persian"
    HELLENISTIC = "Hellenistic"
    PARTHIAN = "Parthian"
    PROTO_ELAMITE = "Proto-Elamite"
    OLD_ELAMITE = "Old Elamite"
    MIDDLE_ELAMITE = "Middle Elamite"
    NEO_ELAMITE = "Neo-Elamite"
    STANDARD_BABYLONIAN = "Standard Babylonian"

    @property
    def abbreviation(self) -> str:
        return ABBREVIATIONS[self]


ABBREVIATIONS: Mapping[Stage, str] = {
    Stage.UR_III: "Ur3",
    Stage.OLD_ASSYRIAN: "OA",
    Stage.OLD_BABYLONIAN: "OB",
    Stage.MIDDLE_BABYLONIAN: "MB",
    Stage.MIDDLE_ASSYRIAN: "MA",
    Stage.HITTITE: "Hit",
    Stage.NEO_ASSYRIAN: "NA",
    Stage.NEO_BABYLONIAN: "NB",
    Stage.LATE_BABYLONIAN: "LB",
    Stage.PERSIAN: "Per",
    Stage.HELLENISTIC: "Hel",
    Stage.PARTHIAN: "Par",
    Stage.UNCERTAIN: "Unc",
    Stage.URUK_IV: "Uruk4",
    Stage.URUK_III_JEMDET_NASR: "JN",
    Stage.ED_I_II: "ED1_2",
    Stage.FARA: "Fara",
    Stage.PRESARGONIC: "PSarg",
    Stage.SARGONIC: "Sarg",
    Stage.STANDARD_BABYLONIAN: "SB",
    Stage.PROTO_ELAMITE: "PElam",
    Stage.OLD_ELAMITE: "OElam",
    Stage.MIDDLE_ELAMITE: "MElam",
    Stage.NEO_ELAMITE: "NElam",
}
