"""ADSI profession -> PLFS denominator crosswalk.

PLFS usual principal status (UPS) codes follow the standard NSS list:
  11 own-account worker, 12 employer, 21 unpaid family helper, 31 regular wage/salaried,
  41 casual labour in public works, 51 casual labour in other work, 81 unemployed,
  91 attending education, 92 domestic duties only, 93 domestic duties + free collection of goods,
  94 rentier/pensioner/remittance recipient, 95 disabled, 97 others (incl. beggars), 99 children 0-4.
Industry: NIC-2008 2-digit 01-03 = agriculture, forestry and fishing.

Each mapping is a function (status, nic2) -> bool that selects the PLFS persons forming the denominator.
The kill check refits the rate ratio under every alternative in ALTERNATIVES. If the defensible
alternatives move the daily-wage rate ratio enough to erase the finding, the project stops.
"""

AGRI = {1, 2, 3}


def primary_daily_wage(status: int, nic2: int) -> bool:
    """PRIMARY denominator for NCRB 'Daily Wage Earner' (which EXCLUDES agricultural labourers).

    Primary = casual labour (public works or other) outside agriculture, forestry and fishing. This follows
    NCRB's own note that 'Daily Wage Earner' excludes agricultural labourers; the alternatives below test it.
    """
    return status in (41, 51) and nic2 not in AGRI


ALTERNATIVES = {
    # casual labour outside agriculture only
    "casual_nonagri": lambda s, n: s in (41, 51) and n not in AGRI,
    # all casual labour, agriculture included (NCRB coders may not separate agriculture reliably)
    "casual_all": lambda s, n: s in (41, 51),
    # casual non-agricultural labour plus own-account workers outside agriculture (informal 'daily earners')
    "casual_plus_own_account_nonagri": lambda s, n: (s in (41, 51) or s == 11) and n not in AGRI,
}
