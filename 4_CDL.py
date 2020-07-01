# -*- coding: utf-8 -*-
"""
Created on Mon May  6 17:33:06 2019

@author: m429673
"""


#categoriser vebt par CDL
AFRICA = ['ABJ', 'ABV', 'ACC', 'BGF', 'BKO', 'BZV', 'CKY', 'COO', 'CPT', 'DKR', 'DLA', 
          'FIH', 'FNA', 'JNB', 'LAD', 'LBV', 'LFW', 'LOS', 'NDJ', 'NIM', 'NKC', 'NSI',
          'OUA', 'PHC', 'PNR', 'SSG', 'JIB']

APC = ['BKK', 'CAN', 'CGK', 'HKG', 'HND', 'ICN', 'KIX', 'NRT', 'PEK', 'PVG', 'SGN',
       'SIN', 'WUH', 'TPE']

COIDOM = ['CAY', 'FDF', 'PTP', 'RUN']

COIINTL = ['SDQ', 'PUJ', 'HAV', 'TNR', 'SXM', 'MRU']

LATINAMERICA = ['BOG', 'CCS', 'EZE', 'GIG', 'GRU', 'LIM', 'MVD', 'PTY', 'SCL', 'SJO']

MEGI = ['AMM', 'BEY', 'DXB', 'IKA', 'RUH', 'BOM', 'BLR', 'CAI', 'DEL']

NATL = ['ATL', 'BOS', 'CUN', 'DTW', 'IAD', 'IAH', 'JFK', 'LAX', 'MEX', 'MIA', 'MSP',
        'ORD', 'PPT', 'SFO', 'YUL', 'YVR', 'YYZ']

#VBT CDL AFRICA
AFRICA_CDG = dataset.loc[(dataset['FinArrSt'].isin(AFRICA))]
AFRICA_BDL = dataset.loc[(dataset['FinDepSt'].isin(AFRICA))]
AFRICA_VBT = AFRICA_CDG.append(AFRICA_BDL)

#VBT CDL APC
APC_CDG = dataset.loc[(dataset['FinArrSt'].isin(APC))]
APC_BDL = dataset.loc[(dataset['FinDepSt'].isin(APC))]
APC_VBT = APC_CDG.append(APC_BDL)

#VBT CDL COIDOM
COIDOM_CDG = dataset.loc[(dataset['FinArrSt'].isin(COIDOM))]
COIDOM_BDL = dataset.loc[(dataset['FinDepSt'].isin(COIDOM))]
COIDOM_VBT = COIDOM_CDG.append(COIDOM_BDL)

#VBT CDL COIINTL
COIINTL_CDG = dataset.loc[(dataset['FinArrSt'].isin(COIINTL))]
COIINTL_BDL = dataset.loc[(dataset['FinDepSt'].isin(COIINTL))]
COIINTL_VBT = COIINTL_CDG.append(COIINTL_BDL)

#VBT CDL LATIN AMERICA
LATINAMERICA_CDG = dataset.loc[(dataset['FinArrSt'].isin(LATINAMERICA))]
LATINAMERICA_BDL = dataset.loc[(dataset['FinDepSt'].isin(LATINAMERICA))]
LATINAMERICA_VBT = LATINAMERICA_CDG.append(LATINAMERICA_BDL)

#VBT CDL MEGI
MEGI_CDG = dataset.loc[(dataset['FinArrSt'].isin(MEGI))]
MEGI_BDL = dataset.loc[(dataset['FinDepSt'].isin(MEGI))]
MEGI_VBT = MEGI_CDG.append(MEGI_BDL)

#VBT CDL NATL
NATL_CDG = dataset.loc[(dataset['FinArrSt'].isin(NATL))]
NATL_BDL = dataset.loc[(dataset['FinDepSt'].isin(NATL))]
NATL_VBT = NATL_CDG.append(NATL_BDL)