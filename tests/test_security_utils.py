import pandas as pd

from security_utils import prepare_csv_export


def test_csv_export_neutralizes_spreadsheet_formulas_without_mutating_source():
    source = pd.DataFrame(
        {
            "Narrative": ["=1+1", "  @SUM(A1:A2)", "+cmd", "-2", "normal complaint"],
            "Count": [1, 2, 3, 4, 5],
        }
    )

    exported = prepare_csv_export(source)

    assert exported["Narrative"].tolist() == [
        "'=1+1",
        "'  @SUM(A1:A2)",
        "'+cmd",
        "'-2",
        "normal complaint",
    ]
    assert source.loc[0, "Narrative"] == "=1+1"
    assert exported["Count"].tolist() == [1, 2, 3, 4, 5]
