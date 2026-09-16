"""Кеш допустим только при доказуемо локальных ссылках формулы."""
import pytest

from laim_basket.reading.formulas import is_row_local_formula


@pytest.mark.parametrize('formula', ['=IF(SUM(A2:B2)=2,1,0)', '=A2+B2',
    '=AVERAGE($A$2:$B2)', '=IF(A2="A1",1,0)', '=1', '=IF(A2>0;1;0)'])
def test_row_local_formulas(formula):
    assert is_row_local_formula(formula, 2)


@pytest.mark.parametrize('formula', ['=SUM(A1:A2)', '=A1+A2', '=SUM(A:A)',
    '=Other!A2', "='Other sheet'!A2", '=[external.xlsx]Sheet1!A2',
    '=SUM(1:2)', '=OFFSET(A2,1,0)', '=INDIRECT("A3")', '=named_range',
    '=SUM(Table1[score])', '=RAND()', '=A2#', '=SUM(A2:A3)', '=IF(A2>1,B2,B3)'])
def test_nonlocal_or_unprovable_formulas(formula):
    assert not is_row_local_formula(formula, 2)
