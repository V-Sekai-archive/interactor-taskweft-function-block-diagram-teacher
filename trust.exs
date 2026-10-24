# The trusted data sources, derived from the ReBAC tuples in OpenBao KV
# `relationships/` (RFD66606-2200, verb `trusts`). Bao is authoritative; this file is
# the derived copy the writer and the publisher read, and the anti-entropy pass
# compares the two. It is read as an AST and pattern-matched, never evaluated.
#
# A code repository is not a data source: forks are tooling, and a placed fixture
# input whose bytes no published row carries is not listed here either.
%{
  subject: "magi-16739d",
  verb: "trusts",
  objects: [
    "huggingface.co/chibifire",
    "v-sekai-fabric/weftspun-keypoint",
    "chibifire/starforged-std-3001-appendix-e",
    # Operator, 2026-09-08, naming item 4's photo leg: "item 4's photo leg exists
    # because in hugging face we have issai/Speaking_Faces". CC BY 4.0 on the
    # project page, MIT for the code. The matching Bao tuple is added when this
    # desk's policy grants a write under `relationships/`.
    "issai/Speaking_Faces"
  ]
}
