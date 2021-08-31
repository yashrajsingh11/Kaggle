# Linear Regression Model to Predict 
# Housing Prices in the City
# Without using any External 
# Machine Learning Library

# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data-preprocessing

train_df = pd.read_csv("train.csv")
train_df.info()

train_df.loc[train_df['MSZoning'] == 'C (all)', 'MSZoning'] = 1
train_df.loc[train_df['MSZoning'] == 'RM', 'MSZoning'] = 2
train_df.loc[train_df['MSZoning'] == 'RH', 'MSZoning'] = 3
train_df.loc[train_df['MSZoning'] == 'RL', 'MSZoning'] = 4
train_df.loc[train_df['MSZoning'] == 'FV', 'MSZoning'] = 5
train_df['MSZoning'] = train_df['MSZoning'].astype('int')

train_df.loc[train_df['Street'] == 'Grvl', 'Street'] = 1
train_df.loc[train_df['Street'] == 'Pave', 'Street'] = 2
train_df['Street'] = train_df['Street'].astype('int')

train_df.loc[train_df['Alley'] == 'Grvl', 'Alley'] = 0
train_df.loc[train_df['Alley'] == 'Pave', 'Alley'] = 1
train_df['Alley'].replace(np.NaN, 2, inplace = True)
train_df['Alley'] = train_df['Alley'].astype('int')

train_df.loc[train_df['LotShape'] == 'Reg', 'LotShape'] = 1
train_df.loc[train_df['LotShape'] == 'IR1', 'LotShape'] = 2
train_df.loc[train_df['LotShape'] == 'IR3', 'LotShape'] = 3
train_df.loc[train_df['LotShape'] == 'IR2', 'LotShape'] = 4
train_df['LotShape'] = train_df['LotShape'].astype('int')

train_df.loc[train_df['LandContour'] == 'Bnk', 'LandContour'] = 1
train_df.loc[train_df['LandContour'] == 'Lvl', 'LandContour'] = 2
train_df.loc[train_df['LandContour'] == 'Low', 'LandContour'] = 3
train_df.loc[train_df['LandContour'] == 'HLS', 'LandContour'] = 4
train_df['LandContour'] = train_df['LandContour'].astype('int')

train_df.loc[train_df['LotConfig'] == 'Inside', 'LotConfig'] = 1
train_df.loc[train_df['LotConfig'] == 'FR2', 'LotConfig'] = 2
train_df.loc[train_df['LotConfig'] == 'Corner', 'LotConfig'] = 3
train_df.loc[train_df['LotConfig'] == 'FR3', 'LotConfig'] = 4
train_df.loc[train_df['LotConfig'] == 'CulDSac', 'LotConfig'] = 5
train_df['LotConfig'] = train_df['LotConfig'].astype('int')

train_df.loc[train_df['LandSlope'] == 'Gtl', 'LandSlope'] = 1
train_df.loc[train_df['LandSlope'] == 'Mod', 'LandSlope'] = 2
train_df.loc[train_df['LandSlope'] == 'Sev', 'LandSlope'] = 3
train_df['LandSlope'] = train_df['LandSlope'].astype('int')

train_df.loc[train_df['Neighborhood'] == 'MeadowV', 'Neighborhood'] = 1
train_df.loc[train_df['Neighborhood'] == 'IDOTRR', 'Neighborhood'] = 2
train_df.loc[train_df['Neighborhood'] == 'BrDale', 'Neighborhood'] = 3
train_df.loc[train_df['Neighborhood'] == 'BrkSide', 'Neighborhood'] = 4
train_df.loc[train_df['Neighborhood'] == 'OldTown', 'Neighborhood'] = 5
train_df.loc[train_df['Neighborhood'] == 'Edwards', 'Neighborhood'] = 6
train_df.loc[train_df['Neighborhood'] == 'Sawyer', 'Neighborhood'] = 7
train_df.loc[train_df['Neighborhood'] == 'Blueste', 'Neighborhood'] = 8
train_df.loc[train_df['Neighborhood'] == 'SWISU', 'Neighborhood'] = 9
train_df.loc[train_df['Neighborhood'] == 'NPkVill', 'Neighborhood'] = 10
train_df.loc[train_df['Neighborhood'] == 'NAmes', 'Neighborhood'] = 11
train_df.loc[train_df['Neighborhood'] == 'Mitchel', 'Neighborhood'] = 12
train_df.loc[train_df['Neighborhood'] == 'SawyerW', 'Neighborhood'] = 13
train_df.loc[train_df['Neighborhood'] == 'NWAmes', 'Neighborhood'] = 14
train_df.loc[train_df['Neighborhood'] == 'Gilbert', 'Neighborhood'] = 15
train_df.loc[train_df['Neighborhood'] == 'Blmngtn', 'Neighborhood'] = 16
train_df.loc[train_df['Neighborhood'] == 'CollgCr', 'Neighborhood'] = 17
train_df.loc[train_df['Neighborhood'] == 'Crawfor', 'Neighborhood'] = 18
train_df.loc[train_df['Neighborhood'] == 'ClearCr', 'Neighborhood'] = 19
train_df.loc[train_df['Neighborhood'] == 'Somerst', 'Neighborhood'] = 20
train_df.loc[train_df['Neighborhood'] == 'Veenker', 'Neighborhood'] = 21
train_df.loc[train_df['Neighborhood'] == 'Timber', 'Neighborhood'] = 22
train_df.loc[train_df['Neighborhood'] == 'StoneBr', 'Neighborhood'] = 23
train_df.loc[train_df['Neighborhood'] == 'NridgHt', 'Neighborhood'] = 24
train_df.loc[train_df['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 25
train_df['Neighborhood'] = train_df['Neighborhood'].astype('int')

train_df.loc[train_df['Condition1'] == 'Artery', 'Condition1'] = 1
train_df.loc[train_df['Condition1'] == 'RRAe', 'Condition1'] = 2
train_df.loc[train_df['Condition1'] == 'Feedr', 'Condition1'] = 3
train_df.loc[train_df['Condition1'] == 'RRAn', 'Condition1'] = 4
train_df.loc[train_df['Condition1'] == 'Norm', 'Condition1'] = 5
train_df.loc[train_df['Condition1'] == 'RRNe', 'Condition1'] = 6
train_df.loc[train_df['Condition1'] == 'RRNn', 'Condition1'] = 7
train_df.loc[train_df['Condition1'] == 'PosN', 'Condition1'] = 8
train_df.loc[train_df['Condition1'] == 'PosA', 'Condition1'] = 9
train_df['Condition1'] = train_df['Condition1'].astype('int')

train_df.loc[train_df['Condition2'] == 'RRNn', 'Condition2'] = 1
train_df.loc[train_df['Condition2'] == 'Artery', 'Condition2'] = 2
train_df.loc[train_df['Condition2'] == 'Feedr', 'Condition2'] = 3
train_df.loc[train_df['Condition2'] == 'RRAn', 'Condition2'] = 4
train_df.loc[train_df['Condition2'] == 'Norm', 'Condition2'] = 5
train_df.loc[train_df['Condition2'] == 'RRAe', 'Condition2'] = 6
train_df.loc[train_df['Condition2'] == 'RRNe', 'Condition2'] = 7
train_df.loc[train_df['Condition2'] == 'PosN', 'Condition2'] = 8
train_df.loc[train_df['Condition2'] == 'PosA', 'Condition2'] = 9
train_df['Condition2'] = train_df['Condition2'].astype('int')

train_df.loc[train_df['BldgType'] == '2fmCon', 'BldgType'] = 1
train_df.loc[train_df['BldgType'] == 'Duplex', 'BldgType'] = 2
train_df.loc[train_df['BldgType'] == 'Twnhs', 'BldgType'] = 3
train_df.loc[train_df['BldgType'] == 'TwnhsE', 'BldgType'] = 4
train_df.loc[train_df['BldgType'] == '1Fam', 'BldgType'] = 5
train_df['BldgType'] = train_df['BldgType'].astype('int')

train_df.loc[train_df['HouseStyle'] == '1.5Unf', 'HouseStyle'] = 1
train_df.loc[train_df['HouseStyle'] == 'SFoyer', 'HouseStyle'] = 2
train_df.loc[train_df['HouseStyle'] == '1.5Fin', 'HouseStyle'] = 3
train_df.loc[train_df['HouseStyle'] == '2.5Unf', 'HouseStyle'] = 4
train_df.loc[train_df['HouseStyle'] == 'SLvl', 'HouseStyle'] = 5
train_df.loc[train_df['HouseStyle'] == '1Story', 'HouseStyle'] = 6
train_df.loc[train_df['HouseStyle'] == '2Story', 'HouseStyle'] = 7
train_df.loc[train_df['HouseStyle'] == '2.5Fin', 'HouseStyle'] = 8
train_df['HouseStyle'] = train_df['HouseStyle'].astype('int')

train_df.loc[train_df['RoofStyle'] == 'Gambrel', 'RoofStyle'] = 1
train_df.loc[train_df['RoofStyle'] == 'Gable', 'RoofStyle'] = 2
train_df.loc[train_df['RoofStyle'] == 'Mansard', 'RoofStyle'] = 3
train_df.loc[train_df['RoofStyle'] == 'Flat', 'RoofStyle'] = 4
train_df.loc[train_df['RoofStyle'] == 'Hip', 'RoofStyle'] = 5
train_df.loc[train_df['RoofStyle'] == 'Shed', 'RoofStyle'] = 6
train_df['RoofStyle'] = train_df['RoofStyle'].astype('int')

train_df.loc[train_df['RoofMatl'] == 'Roll', 'RoofMatl'] = 1
train_df.loc[train_df['RoofMatl'] == 'ClyTile', 'RoofMatl'] = 2
train_df.loc[train_df['RoofMatl'] == 'Metal', 'RoofMatl'] = 3
train_df.loc[train_df['RoofMatl'] == 'CompShg', 'RoofMatl'] = 4
train_df.loc[train_df['RoofMatl'] == 'Tar&Grv', 'RoofMatl'] = 5
train_df.loc[train_df['RoofMatl'] == 'Membran', 'RoofMatl'] = 6
train_df.loc[train_df['RoofMatl'] == 'WdShake', 'RoofMatl'] = 7
train_df.loc[train_df['RoofMatl'] == 'WdShngl', 'RoofMatl'] = 8
train_df['RoofMatl'] = train_df['RoofMatl'].astype('int')

train_df.loc[train_df['Exterior1st'] == 'BrkComm', 'Exterior1st'] = 1
train_df.loc[train_df['Exterior1st'] == 'AsphShn', 'Exterior1st'] = 2
train_df.loc[train_df['Exterior1st'] == 'CBlock', 'Exterior1st'] = 3
train_df.loc[train_df['Exterior1st'] == 'AsbShng', 'Exterior1st'] = 4
train_df.loc[train_df['Exterior1st'] == 'MetalSd', 'Exterior1st'] = 5
train_df.loc[train_df['Exterior1st'] == 'Wd Sdng', 'Exterior1st'] = 6
train_df.loc[train_df['Exterior1st'] == 'WdShing', 'Exterior1st'] = 7
train_df.loc[train_df['Exterior1st'] == 'Stucco', 'Exterior1st'] = 8
train_df.loc[train_df['Exterior1st'] == 'HdBoard', 'Exterior1st'] = 9
train_df.loc[train_df['Exterior1st'] == 'Plywood', 'Exterior1st'] = 10
train_df.loc[train_df['Exterior1st'] == 'BrkFace', 'Exterior1st'] = 11
train_df.loc[train_df['Exterior1st'] == 'VinylSd', 'Exterior1st'] = 12
train_df.loc[train_df['Exterior1st'] == 'CemntBd', 'Exterior1st'] = 13
train_df.loc[train_df['Exterior1st'] == 'Stone', 'Exterior1st'] = 14
train_df.loc[train_df['Exterior1st'] == 'ImStucc', 'Exterior1st'] = 15
train_df['Exterior1st'] = train_df['Exterior1st'].astype('int')

train_df.loc[train_df['Exterior2nd'] == 'CBlock', 'Exterior2nd'] = 1
train_df.loc[train_df['Exterior2nd'] == 'AsbShng', 'Exterior2nd'] = 2
train_df.loc[train_df['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 3
train_df.loc[train_df['Exterior2nd'] == 'AsphShn', 'Exterior2nd'] = 4
train_df.loc[train_df['Exterior2nd'] == 'Wd Sdng', 'Exterior2nd'] = 5
train_df.loc[train_df['Exterior2nd'] == 'MetalSd', 'Exterior2nd'] = 6
train_df.loc[train_df['Exterior2nd'] == 'Stucco', 'Exterior2nd'] = 7
train_df.loc[train_df['Exterior2nd'] == 'Stone', 'Exterior2nd'] = 8
train_df.loc[train_df['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 9
train_df.loc[train_df['Exterior2nd'] == 'Plywood', 'Exterior2nd'] = 10
train_df.loc[train_df['Exterior2nd'] == 'HdBoard', 'Exterior2nd'] = 11
train_df.loc[train_df['Exterior2nd'] == 'BrkFace', 'Exterior2nd'] = 12
train_df.loc[train_df['Exterior2nd'] == 'VinylSd', 'Exterior2nd'] = 13
train_df.loc[train_df['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 14
train_df.loc[train_df['Exterior2nd'] == 'ImStucc', 'Exterior2nd'] = 15
train_df.loc[train_df['Exterior2nd'] == 'Other', 'Exterior2nd'] = 16
train_df['Exterior2nd'] = train_df['Exterior2nd'].astype('int')

train_df.loc[train_df['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 1
train_df['MasVnrType'].replace(np.NaN, 2, inplace = True)
train_df.loc[train_df['MasVnrType'] == 'None', 'MasVnrType'] = 2
train_df.loc[train_df['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 3
train_df.loc[train_df['MasVnrType'] == 'CBlock', 'MasVnrType'] = 4
train_df.loc[train_df['MasVnrType'] == 'Stone', 'MasVnrType'] = 5
train_df['MasVnrType'] = train_df['MasVnrType'].astype('int')

train_df.loc[train_df['ExterQual'] == 'Fa', 'ExterQual'] = 1
train_df.loc[train_df['ExterQual'] == 'TA', 'ExterQual'] = 2
train_df.loc[train_df['ExterQual'] == 'Gd', 'ExterQual'] = 3
train_df.loc[train_df['ExterQual'] == 'Ex', 'ExterQual'] = 4
train_df['ExterQual'] = train_df['ExterQual'].astype('int')

train_df.loc[train_df['ExterCond'] == 'Po', 'ExterCond'] = 1
train_df.loc[train_df['ExterCond'] == 'Fa', 'ExterCond'] = 2
train_df.loc[train_df['ExterCond'] == 'TA', 'ExterCond'] = 3
train_df.loc[train_df['ExterCond'] == 'Gd', 'ExterCond'] = 4
train_df.loc[train_df['ExterCond'] == 'Ex', 'ExterCond'] = 5
train_df['ExterCond'] = train_df['ExterCond'].astype('int')

train_df.loc[train_df['Foundation'] == 'Slab', 'Foundation'] = 1
train_df.loc[train_df['Foundation'] == 'BrkTil', 'Foundation'] = 2
train_df.loc[train_df['Foundation'] == 'Stone', 'Foundation'] = 4
train_df.loc[train_df['Foundation'] == 'Wood', 'Foundation'] = 5
train_df.loc[train_df['Foundation'] == 'CBlock', 'Foundation'] = 3
train_df.loc[train_df['Foundation'] == 'PConc', 'Foundation'] = 6
train_df['Foundation'] = train_df['Foundation'].astype('int')

train_df['BsmtQual'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['BsmtQual'] == 'Fa', 'BsmtQual'] = 1
train_df.loc[train_df['BsmtQual'] == 'TA', 'BsmtQual'] = 2
train_df.loc[train_df['BsmtQual'] == 'Gd', 'BsmtQual'] = 3
train_df.loc[train_df['BsmtQual'] == 'Ex', 'BsmtQual'] = 4
train_df['BsmtQual'] = train_df['BsmtQual'].astype('int')

train_df['BsmtCond'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['BsmtCond'] == 'Po', 'BsmtCond'] = 1
train_df.loc[train_df['BsmtCond'] == 'Fa', 'BsmtCond'] = 2
train_df.loc[train_df['BsmtCond'] == 'TA', 'BsmtCond'] = 3
train_df.loc[train_df['BsmtCond'] == 'Gd', 'BsmtCond'] = 4
train_df['BsmtCond'] = train_df['BsmtCond'].astype('int')

train_df['BsmtExposure'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['BsmtExposure'] == 'No', 'BsmtExposure'] = 1
train_df.loc[train_df['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 2
train_df.loc[train_df['BsmtExposure'] == 'Av', 'BsmtExposure'] = 3
train_df.loc[train_df['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 4
train_df['BsmtExposure'] = train_df['BsmtExposure'].astype('int')

train_df['BsmtFinType1'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 1
train_df.loc[train_df['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 2
train_df.loc[train_df['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 3
train_df.loc[train_df['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 4
train_df.loc[train_df['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 5
train_df.loc[train_df['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 6
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].astype('int')

train_df['BsmtFinType2'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 1
train_df.loc[train_df['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 2
train_df.loc[train_df['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 3
train_df.loc[train_df['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 4
train_df.loc[train_df['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 5
train_df.loc[train_df['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 6
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].astype('int')

train_df.loc[train_df['Heating'] == 'Floor', 'Heating'] = 1
train_df.loc[train_df['Heating'] == 'Grav', 'Heating'] = 2
train_df.loc[train_df['Heating'] == 'Wall', 'Heating'] = 3
train_df.loc[train_df['Heating'] == 'OthW', 'Heating'] = 4
train_df.loc[train_df['Heating'] == 'GasW', 'Heating'] = 5
train_df.loc[train_df['Heating'] == 'GasA', 'Heating'] = 6
train_df['Heating'] = train_df['Heating'].astype('int')

train_df.loc[train_df['HeatingQC'] == 'Po', 'HeatingQC'] = 1
train_df.loc[train_df['HeatingQC'] == 'Fa', 'HeatingQC'] = 2
train_df.loc[train_df['HeatingQC'] == 'TA', 'HeatingQC'] = 3
train_df.loc[train_df['HeatingQC'] == 'Gd', 'HeatingQC'] = 4
train_df.loc[train_df['HeatingQC'] == 'Ex', 'HeatingQC'] = 5
train_df['HeatingQC'] = train_df['HeatingQC'].astype('int')

train_df.loc[train_df['CentralAir'] == 'N', 'CentralAir'] = 0
train_df.loc[train_df['CentralAir'] == 'Y', 'CentralAir'] = 1
train_df['CentralAir'] = train_df['CentralAir'].astype('int')

train_df.loc[train_df['Electrical'] == 'Mix', 'Electrical'] = 1
train_df.loc[train_df['Electrical'] == 'FuseP', 'Electrical'] = 2
train_df.loc[train_df['Electrical'] == 'FuseF', 'Electrical'] = 3
train_df.loc[train_df['Electrical'] == 'FuseA', 'Electrical'] = 4
train_df.loc[train_df['Electrical'] == 'SBrkr', 'Electrical'] = 5
train_df['Electrical'].replace(np.NaN, 5, inplace = True)
train_df['Electrical'] = train_df['Electrical'].astype('int')

train_df.loc[train_df['KitchenQual'] == 'Fa', 'KitchenQual'] = 1
train_df.loc[train_df['KitchenQual'] == 'TA', 'KitchenQual'] = 2
train_df.loc[train_df['KitchenQual'] == 'Gd', 'KitchenQual'] = 3
train_df.loc[train_df['KitchenQual'] == 'Ex', 'KitchenQual'] = 4
train_df['KitchenQual'] = train_df['KitchenQual'].astype('int')

train_df.loc[train_df['Functional'] == 'Maj2', 'Functional'] = 1
train_df.loc[train_df['Functional'] == 'Sev', 'Functional'] = 2
train_df.loc[train_df['Functional'] == 'Min2', 'Functional'] = 3
train_df.loc[train_df['Functional'] == 'Min1', 'Functional'] = 4
train_df.loc[train_df['Functional'] == 'Maj1', 'Functional'] = 5
train_df.loc[train_df['Functional'] == 'Mod', 'Functional'] = 6
train_df.loc[train_df['Functional'] == 'Typ', 'Functional'] = 7
train_df['Functional'] = train_df['Functional'].astype('int')

train_df['FireplaceQu'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['FireplaceQu'] == 'Po', 'FireplaceQu'] = 1
train_df.loc[train_df['FireplaceQu'] == 'Fa', 'FireplaceQu'] = 2
train_df.loc[train_df['FireplaceQu'] == 'TA', 'FireplaceQu'] = 3
train_df.loc[train_df['FireplaceQu'] == 'Gd', 'FireplaceQu'] = 4
train_df.loc[train_df['FireplaceQu'] == 'Ex', 'FireplaceQu'] = 5
train_df['FireplaceQu'] = train_df['FireplaceQu'].astype('int')

train_df['GarageType'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['GarageType'] == 'CarPort', 'GarageType'] = 1
train_df.loc[train_df['GarageType'] == 'Detchd', 'GarageType'] = 2
train_df.loc[train_df['GarageType'] == '2Types', 'GarageType'] = 3
train_df.loc[train_df['GarageType'] == 'Basment', 'GarageType'] = 4
train_df.loc[train_df['GarageType'] == 'Attchd', 'GarageType'] = 5
train_df.loc[train_df['GarageType'] == 'BuiltIn', 'GarageType'] = 6
train_df['GarageType'] = train_df['GarageType'].astype('int')

train_df['GarageFinish'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['GarageFinish'] == 'Unf', 'GarageFinish'] = 1
train_df.loc[train_df['GarageFinish'] == 'RFn', 'GarageFinish'] = 2
train_df.loc[train_df['GarageFinish'] == 'Fin', 'GarageFinish'] = 3
train_df['GarageFinish'] = train_df['GarageFinish'].astype('int')

train_df['GarageQual'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['GarageQual'] == 'Po', 'GarageQual'] = 1
train_df.loc[train_df['GarageQual'] == 'Fa', 'GarageQual'] = 2
train_df.loc[train_df['GarageQual'] == 'TA', 'GarageQual'] = 3
train_df.loc[train_df['GarageQual'] == 'Gd', 'GarageQual'] = 4
train_df.loc[train_df['GarageQual'] == 'Ex', 'GarageQual'] = 5
train_df['GarageQual'] = train_df['GarageQual'].astype('int')

train_df['GarageCond'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['GarageCond'] == 'Po', 'GarageCond'] = 1
train_df.loc[train_df['GarageCond'] == 'Fa', 'GarageCond'] = 2
train_df.loc[train_df['GarageCond'] == 'TA', 'GarageCond'] = 3
train_df.loc[train_df['GarageCond'] == 'Gd', 'GarageCond'] = 4
train_df.loc[train_df['GarageCond'] == 'Ex', 'GarageCond'] = 5
train_df['GarageCond'] = train_df['GarageCond'].astype('int')

train_df.loc[train_df['PavedDrive'] == 'N', 'PavedDrive'] = 0
train_df.loc[train_df['PavedDrive'] == 'P', 'PavedDrive'] = 1
train_df.loc[train_df['PavedDrive'] == 'Y', 'PavedDrive'] = 2
train_df['PavedDrive'] = train_df['PavedDrive'].astype('int')

train_df.loc[train_df['PoolArea'] == 0, 'PoolArea'] = 0
train_df.loc[train_df['PoolArea'] != 0, 'PoolArea'] = 1
train_df['PoolArea'] = train_df['PoolArea'].astype('int')

train_df['Fence'].replace(np.NaN, 0, inplace = True)
train_df.loc[train_df['Fence'] == 'MnWw', 'Fence'] = 1
train_df.loc[train_df['Fence'] == 'GdWo', 'Fence'] = 2
train_df.loc[train_df['Fence'] == 'MnPrv', 'Fence'] = 3
train_df.loc[train_df['Fence'] == 'GdPrv', 'Fence'] = 4
train_df['Fence'] = train_df['Fence'].astype('int')

train_df.loc[train_df['MiscFeature'] == 'Othr', 'MiscFeature'] = 1
train_df.loc[train_df['MiscFeature'] == 'Shed', 'MiscFeature'] = 2
train_df.loc[train_df['MiscFeature'] == 'Gar2', 'MiscFeature'] = 3
train_df['MiscFeature'].replace(np.NaN, 4, inplace = True)
train_df.loc[train_df['MiscFeature'] == 'TenC', 'MiscFeature'] = 5
train_df['MiscFeature'] = train_df['MiscFeature'].astype('int')

train_df.loc[train_df['SaleType'] == 'Oth', 'SaleType'] = 1
train_df.loc[train_df['SaleType'] == 'ConLD', 'SaleType'] = 2
train_df.loc[train_df['SaleType'] == 'ConLw', 'SaleType'] = 3
train_df.loc[train_df['SaleType'] == 'COD', 'SaleType'] = 4
train_df.loc[train_df['SaleType'] == 'WD', 'SaleType'] = 5
train_df.loc[train_df['SaleType'] == 'ConLI', 'SaleType'] = 6
train_df.loc[train_df['SaleType'] == 'CWD', 'SaleType'] = 7
train_df.loc[train_df['SaleType'] == 'Con', 'SaleType'] = 8
train_df.loc[train_df['SaleType'] == 'New', 'SaleType'] = 9
train_df['SaleType'] = train_df['SaleType'].astype('int')

train_df.loc[train_df['SaleCondition'] == 'AdjLand', 'SaleCondition'] = 1
train_df.loc[train_df['SaleCondition'] == 'Abnorml', 'SaleCondition'] = 2
train_df.loc[train_df['SaleCondition'] == 'Family', 'SaleCondition'] = 3
train_df.loc[train_df['SaleCondition'] == 'Alloca', 'SaleCondition'] = 4
train_df.loc[train_df['SaleCondition'] == 'Normal', 'SaleCondition'] = 5
train_df.loc[train_df['SaleCondition'] == 'Partial', 'SaleCondition'] = 6
train_df['SaleCondition'] = train_df['SaleCondition'].astype('int')

train_df.loc[train_df['MSSubClass'] == 30, 'MSSubClass'] = 1
train_df.loc[train_df['MSSubClass'] == 180, 'MSSubClass'] = 2
train_df.loc[train_df['MSSubClass'] == 45, 'MSSubClass'] = 3
train_df.loc[train_df['MSSubClass'] == 190, 'MSSubClass'] = 4
train_df.loc[train_df['MSSubClass'] == 90, 'MSSubClass'] = 5
train_df.loc[train_df['MSSubClass'] == 160, 'MSSubClass'] = 6
train_df.loc[train_df['MSSubClass'] == 50, 'MSSubClass'] = 7
train_df.loc[train_df['MSSubClass'] == 85, 'MSSubClass'] = 8
train_df.loc[train_df['MSSubClass'] == 40, 'MSSubClass'] = 9
train_df.loc[train_df['MSSubClass'] == 70, 'MSSubClass'] = 10
train_df.loc[train_df['MSSubClass'] == 80, 'MSSubClass'] = 11
train_df.loc[train_df['MSSubClass'] == 20, 'MSSubClass'] = 12
train_df.loc[train_df['MSSubClass'] == 75, 'MSSubClass'] = 13
train_df.loc[train_df['MSSubClass'] == 120, 'MSSubClass'] = 14
train_df.loc[train_df['MSSubClass'] == 150, 'MSSubClass'] = 15
train_df.loc[train_df['MSSubClass'] == 60, 'MSSubClass'] = 16
train_df['MSSubClass'] = train_df['MSSubClass'].astype('int')

garageYearBuiltMean = train_df['GarageYrBlt'].mean()
train_df['GarageYrBlt'].replace(np.NaN, garageYearBuiltMean, inplace = True)

MasVnrAreaMean = train_df['MasVnrArea'].mean()
train_df['MasVnrArea'].replace(np.NaN, MasVnrAreaMean, inplace = True)

lotFrontageMean = train_df['LotFrontage'].mean()
train_df['LotFrontage'].replace(np.NaN, lotFrontageMean, inplace = True)

# Data Analysis

pd.options.display.max_rows = None
corr = train_df.corr()
print(corr['SalePrice'].sort_values(ascending=False))
    
#plt.subplot(2, 3, 1)
#plt.plot((train_df['SalePrice'].groupby(train_df["MSSubClass"])).mean())
#plt.subplot(2, 3, 2)
#plt.plot((train_df['SalePrice'].groupby(train_df["MSZoning"])).mean())
#plt.subplot(2, 3, 3)
#plt.plot((train_df['SalePrice'].groupby(train_df["LotShape"])).mean())
#plt.subplot(2, 3, 4)
#plt.plot((train_df['SalePrice'].groupby(train_df["LandContour"])).mean())
#plt.subplot(2, 3, 5)
#plt.plot((train_df['SalePrice'].groupby(train_df["Neighborhood"])).mean())
#plt.subplot(2, 3, 6)
#plt.plot((train_df['SalePrice'].groupby(train_df["BldgType"])).mean())

# Feature Selection

arr_train = train_df.to_numpy()

train_OverallQuality = np.array(arr_train[:,17], np.int8)
train_size = train_OverallQuality.size

train_GroundLivingArea = np.array(arr_train[:,46], np.float32)
train_GroundLivingArea = train_GroundLivingArea/100

train_Neighborhood = np.array(arr_train[:,12], np.int8)

train_ExteriorQuality = np.array(arr_train[:,27], np.int8)

train_KitchenQuality = np.array(arr_train[:,53], np.int8)

train_GarageCars = np.array(arr_train[:,61], np.int8)

train_GarageArea = np.array(arr_train[:,62], np.float32)
train_GarageArea = train_GarageArea/100

train_BasementQuality = np.array(arr_train[:,30], np.int8)

train_TotalBasementSF = np.array(arr_train[:,38], np.float32)
train_TotalBasementSF = train_TotalBasementSF/100

train_1stFlrSF = np.array(arr_train[:,43], np.float32)
train_1stFlrSF = train_1stFlrSF/100

train_FullBath = np.array(arr_train[:,49], np.int8)

train_GarageFinish = np.array(arr_train[:,60], np.int8)

train_TotalRoomsAboveGround = np.array(arr_train[:,54], np.int8)

train_FireplaceQuality = np.array(arr_train[:,57], np.int8)

train_YearRemodeled = np.array(arr_train[:,20], np.float32)
train_YearRemodeled = train_YearRemodeled/100

train_Foundation = np.array(arr_train[:,29], np.int8)

train_GarageType = np.array(arr_train[:,58], np.int8)

train_GarageYearBuilt = np.array(arr_train[:,59], np.float32)
train_GarageYearBuilt = train_GarageYearBuilt/100

train_MSSubClass = np.array(arr_train[:,1], np.int8)

train_MasVnrArea = np.array(arr_train[:,26], np.float32)
train_MasVnrArea = train_MasVnrArea/100

train_Fireplaces = np.array(arr_train[:,56], np.int8)

train_HeatingQuality = np.array(arr_train[:,40], np.int8)

train_MasVnrType = np.array(arr_train[:,25], np.int8)

train_BasementFinSF1 = np.array(arr_train[:,34], np.float32)
train_BasementFinSF1 = train_BasementFinSF1/100

train_BasementExposure = np.array(arr_train[:,32], np.int8)

train_ExteriorFirst = np.array(arr_train[:,23], np.int8)

train_SaleType = np.array(arr_train[:,78], np.int8)

train_ExteriorSecond = np.array(arr_train[:,24], np.int8)

train_LotFrontage = np.array(arr_train[:,23], np.float32)
train_LotFrontage = train_LotFrontage/10

train_MSZoning = np.array(arr_train[:,2], np.int8)

train_WoodDeckSF = np.array(arr_train[:,66], np.float32)
train_WoodDeckSF = train_WoodDeckSF/100

train_2ndFlrSF = np.array(arr_train[:,44], np.float32)
train_2ndFlrSF = train_2ndFlrSF/100

train_OpenPorchSF = np.array(arr_train[:,67], np.float32)
train_OpenPorchSF = train_OpenPorchSF/100

train_BsmtFinType1 = np.array(arr_train[:,33], np.int8)

train_HalfBath = np.array(arr_train[:,50], np.int8)

train_GarageQual = np.array(arr_train[:,63], np.int8)

train_HouseStyle = np.array(arr_train[:,16], np.int8)

train_LotShape = np.array(arr_train[:,7], np.int8)

train_LotArea = np.array(arr_train[:,4], np.float32)
train_LotArea = train_LotArea/10000

train_GarageCond = np.array(arr_train[:,64], np.int8)

train_CentralAir = np.array(arr_train[:,41], np.int8)

train_RoofStyle = np.array(arr_train[:,21], np.int8)

train_SaleCondition = np.array(arr_train[:,79], np.int8)

train_Electrical = np.array(arr_train[:,42], np.int8)

train_PavedDrive = np.array(arr_train[:,65], np.int8)

train_BsmtFullBath = np.array(arr_train[:,47], np.int8)

train_BsmtUnfSF = np.array(arr_train[:,37], np.float32)
train_BsmtUnfSF = train_BsmtUnfSF/100

train_BsmtCond = np.array(arr_train[:,31], np.int8)

train_BldgType = np.array(arr_train[:,15], np.int8)

train_Condition1 = np.array(arr_train[:,13], np.int8)

train_BedroomAbvGr = np.array(arr_train[:,51], np.int8)

train_RoofMatl = np.array(arr_train[:,22], np.int8)

train_LandContour = np.array(arr_train[:,8], np.int8)

train_Alley = np.array(arr_train[:,6], np.int8)

train_Functional = np.array(arr_train[:,5], np.int8)

train_LotConfig = np.array(arr_train[:,10], np.int8)

train_Heating = np.array(arr_train[:,39], np.int8)

train_ScreenPorch = np.array(arr_train[:,70], np.float32)
train_ScreenPorch = train_ScreenPorch/100

train_Condition2 = np.array(arr_train[:,14], np.int8)

train_PoolArea = np.array(arr_train[:,71], np.int8)

train_MiscFeature = np.array(arr_train[:,74], np.int8)

train_LandSlope = np.array(arr_train[:,11], np.int8)

train_OverallCond = np.array(arr_train[:,18], np.int8)

train_EnclosedPorch = np.array(arr_train[:,68], np.float32)
train_EnclosedPorch = train_EnclosedPorch/100

train_KitchenAbvGr = np.array(arr_train[:,52], np.int8)

train_Fence = np.array(arr_train[:,73], np.int8)

train_X = np.ones(train_size, np.int8)

train_SalePrice = np.array(arr_train[:,80], np.int64)

# Model

theta = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
alpha = 0.0005

for j in range(0,9000):
    feature = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 
    print(j)
    for i in range(0, train_size):
        t = train_X[i]*theta[0] + train_OverallQuality[i]*theta[1] + train_GroundLivingArea[i]*theta[2] + train_Neighborhood[i]*theta[3] + train_ExteriorQuality[i]*theta[4] + train_KitchenQuality[i]*theta[5] + train_GarageCars[i]*theta[6] + train_GarageArea[i]*theta[7] + train_BasementQuality[i]*theta[8] + train_TotalBasementSF[i]*theta[9] + train_1stFlrSF[i]*theta[10] + train_FullBath[i]*theta[11] + train_GarageFinish[i]*theta[12] + train_TotalRoomsAboveGround[i]*theta[13] + train_FireplaceQuality[i]*theta[14] + train_YearRemodeled[i]*theta[15] + train_Foundation[i]*theta[16] + train_GarageType[i]*theta[17] + train_GarageYearBuilt[i]*theta[18] + train_MSSubClass[i]*theta[19] + train_MasVnrArea[i]*theta[20] + train_Fireplaces[i]*theta[21] + train_HeatingQuality[i]*theta[22] + train_MasVnrType[i]*theta[23] + train_BasementFinSF1[i]*theta[24] + train_BasementExposure[i]*theta[25] + train_ExteriorFirst[i]*theta[26] + train_SaleType[i]*theta[27] + train_ExteriorSecond[i]*theta[28] + train_LotFrontage[i]*theta[29] + train_MSZoning[i]*theta[30] + train_WoodDeckSF[i]*theta[31] + train_2ndFlrSF[i]*theta[32] + train_OpenPorchSF[i]*theta[33] + train_BsmtFinType1[i]*theta[34] + train_HalfBath[i]*theta[35] + train_GarageQual[i]*theta[36] + train_HouseStyle[i]*theta[37] + train_LotShape[i]*theta[38] + train_LotArea[i]*theta[39] + train_GarageCond[i]*theta[40] + train_CentralAir[i]*theta[41] + train_RoofStyle[i]*theta[42] + train_SaleCondition[i]*theta[43] + train_Electrical[i]*theta[44] + train_PavedDrive[i]*theta[45] + train_BsmtFullBath[i]*theta[46] + train_BsmtUnfSF[i]*theta[47] + train_BsmtCond[i]*theta[48] + train_BldgType[i]*theta[49] + train_Condition1[i]*theta[50] + train_BedroomAbvGr[i]*theta[51] + train_RoofMatl[i]*theta[52] + train_LandContour[i]*theta[53] + train_Alley[i]*theta[54] + train_Functional[i]*theta[55] + train_LotConfig[i]*theta[56] + train_Heating[i]*theta[57] + train_ScreenPorch[i]*theta[58] + train_Condition2[i]*theta[59] + train_PoolArea[i]*theta[60] + train_MiscFeature[i]*theta[61] + train_LandSlope[i]*theta[62] + train_OverallCond[i]*theta[63] + train_EnclosedPorch[i]*theta[64] + train_KitchenAbvGr[i]*theta[65] + train_Fence[i]*theta[66]                                                                                                                                            
        t = (t - train_SalePrice[i])
        feature[0] = feature[0] + t*train_X[i]
        feature[1] = feature[1] + t*train_OverallQuality[i]
        feature[2] = feature[2] + t*train_GroundLivingArea[i]
        feature[3] = feature[3] + t*train_Neighborhood[i]
        feature[4] = feature[4] + t*train_ExteriorQuality[i]
        feature[5] = feature[5] + t*train_KitchenQuality[i]
        feature[6] = feature[6] + t*train_GarageCars[i]
        feature[7] = feature[7] + t*train_GarageArea[i]
        feature[8] = feature[8] + t*train_BasementQuality[i]
        feature[9] = feature[9] + t*train_TotalBasementSF[i]
        feature[10] = feature[10] + t*train_1stFlrSF[i]
        feature[11] = feature[11] + t*train_FullBath[i]
        feature[12] = feature[12] + t*train_GarageFinish[i]
        feature[13] = feature[13] + t*train_TotalRoomsAboveGround[i]
        feature[14] = feature[14] + t*train_FireplaceQuality[i]
        feature[15] = feature[15] + t*train_YearRemodeled[i]
        feature[16] = feature[16] + t*train_Foundation[i]
        feature[17] = feature[17] + t*train_GarageType[i]
        feature[18] = feature[18] + t*train_GarageYearBuilt[i]
        feature[19] = feature[19] + t*train_MSSubClass[i]
        feature[20] = feature[20] + t*train_MasVnrArea[i]
        feature[21] = feature[21] + t*train_Fireplaces[i]
        feature[22] = feature[22] + t*train_HeatingQuality[i]
        feature[23] = feature[23] + t*train_MasVnrType[i]
        feature[24] = feature[24] + t*train_BasementFinSF1[i]
        feature[25] = feature[25] + t*train_BasementExposure[i]
        feature[26] = feature[26] + t*train_ExteriorFirst[i]
        feature[27] = feature[27] + t*train_SaleType[i]
        feature[28] = feature[28] + t*train_ExteriorSecond[i]
        feature[29] = feature[29] + t*train_LotFrontage[i]
        feature[30] = feature[30] + t*train_MSZoning[i]
        feature[31] = feature[31] + t*train_WoodDeckSF[i]
        feature[32] = feature[32] + t*train_2ndFlrSF[i]
        feature[33] = feature[33] + t*train_OpenPorchSF[i]
        feature[34] = feature[34] + t*train_BsmtFinType1[i]
        feature[35] = feature[35] + t*train_HalfBath[i]
        feature[36] = feature[36] + t*train_GarageQual[i]
        feature[37] = feature[37] + t*train_HouseStyle[i]
        feature[38] = feature[38] + t*train_LotShape[i]
        feature[39] = feature[39] + t*train_LotArea[i]
        feature[40] = feature[40] + t*train_GarageCond[i]
        feature[41] = feature[41] + t*train_CentralAir[i]
        feature[42] = feature[42] + t*train_RoofStyle[i]
        feature[43] = feature[43] + t*train_SaleCondition[i]
        feature[44] = feature[44] + t*train_Electrical[i]
        feature[45] = feature[45] + t*train_PavedDrive[i]
        feature[46] = feature[46] + t*train_BsmtFullBath[i]
        feature[47] = feature[47] + t*train_BsmtUnfSF[i]
        feature[48] = feature[48] + t*train_BsmtCond[i]
        feature[49] = feature[49] + t*train_BldgType[i]
        feature[50] = feature[50] + t*train_Condition1[i]
        feature[51] = feature[51] + t*train_BedroomAbvGr[i]
        feature[52] = feature[52] + t*train_RoofMatl[i]
        feature[53] = feature[53] + t*train_LandContour[i]
        feature[54] = feature[54] + t*train_Alley[i]
        feature[55] = feature[55] + t*train_Functional[i]
        feature[56] = feature[56] + t*train_LotConfig[i]
        feature[57] = feature[57] + t*train_Heating[i]
        feature[58] = feature[58] + t*train_ScreenPorch[i]
        feature[59] = feature[59] + t*train_Condition2[i]
        feature[60] = feature[60] + t*train_PoolArea[i]
        feature[61] = feature[61] + t*train_MiscFeature[i]
        feature[62] = feature[62] + t*train_LandSlope[i]
        feature[63] = feature[63] + t*train_OverallCond[i]
        feature[64] = feature[64] + t*train_EnclosedPorch[i]
        feature[65] = feature[65] + t*train_KitchenAbvGr[i]
        feature[66] = feature[66] + t*train_Fence[i]
    for k in range(0, 67):
        feature[k] = (feature[k] * alpha)/1100
        theta[k] = theta[k] - feature[k]
        print(theta[k])

     
# Predictions

test_df = pd.read_csv('test.csv') 
test_df.info()

test_df.loc[test_df['MSZoning'] == 'C (all)', 'MSZoning'] = 1
test_df.loc[test_df['MSZoning'] == 'RM', 'MSZoning'] = 2
test_df.loc[test_df['MSZoning'] == 'RH', 'MSZoning'] = 3
test_df['MSZoning'].replace(np.NaN, 4, inplace = True)
test_df.loc[test_df['MSZoning'] == 'RL', 'MSZoning'] = 4
test_df.loc[test_df['MSZoning'] == 'FV', 'MSZoning'] = 5
test_df['MSZoning'] = test_df['MSZoning'].astype('int')

test_df.loc[test_df['Street'] == 'Grvl', 'Street'] = 1
test_df.loc[test_df['Street'] == 'Pave', 'Street'] = 2
test_df['Street'] = test_df['Street'].astype('int')

test_df.loc[test_df['Alley'] == 'Grvl', 'Alley'] = 0
test_df.loc[test_df['Alley'] == 'Pave', 'Alley'] = 1
test_df['Alley'].replace(np.NaN, 2, inplace = True)
test_df['Alley'] = test_df['Alley'].astype('int')

test_df.loc[test_df['LotShape'] == 'Reg', 'LotShape'] = 1
test_df.loc[test_df['LotShape'] == 'IR1', 'LotShape'] = 2
test_df.loc[test_df['LotShape'] == 'IR3', 'LotShape'] = 3
test_df.loc[test_df['LotShape'] == 'IR2', 'LotShape'] = 4
test_df['LotShape'] = test_df['LotShape'].astype('int')

test_df.loc[test_df['LandContour'] == 'Bnk', 'LandContour'] = 1
test_df.loc[test_df['LandContour'] == 'Lvl', 'LandContour'] = 2
test_df.loc[test_df['LandContour'] == 'Low', 'LandContour'] = 3
test_df.loc[test_df['LandContour'] == 'HLS', 'LandContour'] = 4
test_df['LandContour'] = test_df['LandContour'].astype('int')

test_df.loc[test_df['LotConfig'] == 'Inside', 'LotConfig'] = 1
test_df.loc[test_df['LotConfig'] == 'FR2', 'LotConfig'] = 2
test_df.loc[test_df['LotConfig'] == 'Corner', 'LotConfig'] = 3
test_df.loc[test_df['LotConfig'] == 'FR3', 'LotConfig'] = 4
test_df.loc[test_df['LotConfig'] == 'CulDSac', 'LotConfig'] = 5
test_df['LotConfig'] = test_df['LotConfig'].astype('int')

test_df.loc[test_df['LandSlope'] == 'Gtl', 'LandSlope'] = 1
test_df.loc[test_df['LandSlope'] == 'Mod', 'LandSlope'] = 2
test_df.loc[test_df['LandSlope'] == 'Sev', 'LandSlope'] = 3
test_df['LandSlope'] = test_df['LandSlope'].astype('int')

test_df.loc[test_df['Neighborhood'] == 'MeadowV', 'Neighborhood'] = 1
test_df.loc[test_df['Neighborhood'] == 'IDOTRR', 'Neighborhood'] = 2
test_df.loc[test_df['Neighborhood'] == 'BrDale', 'Neighborhood'] = 3
test_df.loc[test_df['Neighborhood'] == 'BrkSide', 'Neighborhood'] = 4
test_df.loc[test_df['Neighborhood'] == 'OldTown', 'Neighborhood'] = 5
test_df.loc[test_df['Neighborhood'] == 'Edwards', 'Neighborhood'] = 6
test_df.loc[test_df['Neighborhood'] == 'Sawyer', 'Neighborhood'] = 7
test_df.loc[test_df['Neighborhood'] == 'Blueste', 'Neighborhood'] = 8
test_df.loc[test_df['Neighborhood'] == 'SWISU', 'Neighborhood'] = 9
test_df.loc[test_df['Neighborhood'] == 'NPkVill', 'Neighborhood'] = 10
test_df.loc[test_df['Neighborhood'] == 'NAmes', 'Neighborhood'] = 11
test_df.loc[test_df['Neighborhood'] == 'Mitchel', 'Neighborhood'] = 12
test_df.loc[test_df['Neighborhood'] == 'SawyerW', 'Neighborhood'] = 13
test_df.loc[test_df['Neighborhood'] == 'NWAmes', 'Neighborhood'] = 14
test_df.loc[test_df['Neighborhood'] == 'Gilbert', 'Neighborhood'] = 15
test_df.loc[test_df['Neighborhood'] == 'Blmngtn', 'Neighborhood'] = 16
test_df.loc[test_df['Neighborhood'] == 'CollgCr', 'Neighborhood'] = 17
test_df.loc[test_df['Neighborhood'] == 'Crawfor', 'Neighborhood'] = 18
test_df.loc[test_df['Neighborhood'] == 'ClearCr', 'Neighborhood'] = 19
test_df.loc[test_df['Neighborhood'] == 'Somerst', 'Neighborhood'] = 20
test_df.loc[test_df['Neighborhood'] == 'Veenker', 'Neighborhood'] = 21
test_df.loc[test_df['Neighborhood'] == 'Timber', 'Neighborhood'] = 22
test_df.loc[test_df['Neighborhood'] == 'StoneBr', 'Neighborhood'] = 23
test_df.loc[test_df['Neighborhood'] == 'NridgHt', 'Neighborhood'] = 24
test_df.loc[test_df['Neighborhood'] == 'NoRidge', 'Neighborhood'] = 25
test_df['Neighborhood'] = test_df['Neighborhood'].astype('int')

test_df.loc[test_df['Condition1'] == 'Artery', 'Condition1'] = 1
test_df.loc[test_df['Condition1'] == 'RRAe', 'Condition1'] = 2
test_df.loc[test_df['Condition1'] == 'Feedr', 'Condition1'] = 3
test_df.loc[test_df['Condition1'] == 'RRAn', 'Condition1'] = 4
test_df.loc[test_df['Condition1'] == 'Norm', 'Condition1'] = 5
test_df.loc[test_df['Condition1'] == 'RRNe', 'Condition1'] = 6
test_df.loc[test_df['Condition1'] == 'RRNn', 'Condition1'] = 7
test_df.loc[test_df['Condition1'] == 'PosN', 'Condition1'] = 8
test_df.loc[test_df['Condition1'] == 'PosA', 'Condition1'] = 9
test_df['Condition1'] = test_df['Condition1'].astype('int')

test_df.loc[test_df['Condition2'] == 'RRNn', 'Condition2'] = 1
test_df.loc[test_df['Condition2'] == 'Artery', 'Condition2'] = 2
test_df.loc[test_df['Condition2'] == 'Feedr', 'Condition2'] = 3
test_df.loc[test_df['Condition2'] == 'RRAn', 'Condition2'] = 4
test_df.loc[test_df['Condition2'] == 'Norm', 'Condition2'] = 5
test_df.loc[test_df['Condition2'] == 'RRAe', 'Condition2'] = 6
test_df.loc[test_df['Condition2'] == 'RRNe', 'Condition2'] = 7
test_df.loc[test_df['Condition2'] == 'PosN', 'Condition2'] = 8
test_df.loc[test_df['Condition2'] == 'PosA', 'Condition2'] = 9
test_df['Condition2'] = test_df['Condition2'].astype('int')

test_df.loc[test_df['BldgType'] == '2fmCon', 'BldgType'] = 1
test_df.loc[test_df['BldgType'] == 'Duplex', 'BldgType'] = 2
test_df.loc[test_df['BldgType'] == 'Twnhs', 'BldgType'] = 3
test_df.loc[test_df['BldgType'] == 'TwnhsE', 'BldgType'] = 4
test_df.loc[test_df['BldgType'] == '1Fam', 'BldgType'] = 5
test_df['BldgType'] = test_df['BldgType'].astype('int')

test_df.loc[test_df['HouseStyle'] == '1.5Unf', 'HouseStyle'] = 1
test_df.loc[test_df['HouseStyle'] == 'SFoyer', 'HouseStyle'] = 2
test_df.loc[test_df['HouseStyle'] == '1.5Fin', 'HouseStyle'] = 3
test_df.loc[test_df['HouseStyle'] == '2.5Unf', 'HouseStyle'] = 4
test_df.loc[test_df['HouseStyle'] == 'SLvl', 'HouseStyle'] = 5
test_df.loc[test_df['HouseStyle'] == '1Story', 'HouseStyle'] = 6
test_df.loc[test_df['HouseStyle'] == '2Story', 'HouseStyle'] = 7
test_df.loc[test_df['HouseStyle'] == '2.5Fin', 'HouseStyle'] = 8
test_df['HouseStyle'] = test_df['HouseStyle'].astype('int')

test_df.loc[test_df['RoofStyle'] == 'Gambrel', 'RoofStyle'] = 1
test_df.loc[test_df['RoofStyle'] == 'Gable', 'RoofStyle'] = 2
test_df.loc[test_df['RoofStyle'] == 'Mansard', 'RoofStyle'] = 3
test_df.loc[test_df['RoofStyle'] == 'Flat', 'RoofStyle'] = 4
test_df.loc[test_df['RoofStyle'] == 'Hip', 'RoofStyle'] = 5
test_df.loc[test_df['RoofStyle'] == 'Shed', 'RoofStyle'] = 6
test_df['RoofStyle'] = test_df['RoofStyle'].astype('int')

test_df.loc[test_df['RoofMatl'] == 'Roll', 'RoofMatl'] = 1
test_df.loc[test_df['RoofMatl'] == 'ClyTile', 'RoofMatl'] = 2
test_df.loc[test_df['RoofMatl'] == 'Metal', 'RoofMatl'] = 3
test_df.loc[test_df['RoofMatl'] == 'CompShg', 'RoofMatl'] = 4
test_df.loc[test_df['RoofMatl'] == 'Tar&Grv', 'RoofMatl'] = 5
test_df.loc[test_df['RoofMatl'] == 'Membran', 'RoofMatl'] = 6
test_df.loc[test_df['RoofMatl'] == 'WdShake', 'RoofMatl'] = 7
test_df.loc[test_df['RoofMatl'] == 'WdShngl', 'RoofMatl'] = 8
test_df['RoofMatl'] = test_df['RoofMatl'].astype('int')

test_df.loc[test_df['Exterior1st'] == 'BrkComm', 'Exterior1st'] = 1
test_df.loc[test_df['Exterior1st'] == 'AsphShn', 'Exterior1st'] = 2
test_df.loc[test_df['Exterior1st'] == 'CBlock', 'Exterior1st'] = 3
test_df.loc[test_df['Exterior1st'] == 'AsbShng', 'Exterior1st'] = 4
test_df.loc[test_df['Exterior1st'] == 'MetalSd', 'Exterior1st'] = 5
test_df.loc[test_df['Exterior1st'] == 'Wd Sdng', 'Exterior1st'] = 6
test_df.loc[test_df['Exterior1st'] == 'WdShing', 'Exterior1st'] = 7
test_df.loc[test_df['Exterior1st'] == 'Stucco', 'Exterior1st'] = 8
test_df.loc[test_df['Exterior1st'] == 'HdBoard', 'Exterior1st'] = 9
test_df.loc[test_df['Exterior1st'] == 'Plywood', 'Exterior1st'] = 10
test_df.loc[test_df['Exterior1st'] == 'BrkFace', 'Exterior1st'] = 11
test_df['Exterior1st'].replace(np.NaN, 12, inplace = True)
test_df.loc[test_df['Exterior1st'] == 'VinylSd', 'Exterior1st'] = 12
test_df.loc[test_df['Exterior1st'] == 'CemntBd', 'Exterior1st'] = 13
test_df.loc[test_df['Exterior1st'] == 'Stone', 'Exterior1st'] = 14
test_df.loc[test_df['Exterior1st'] == 'ImStucc', 'Exterior1st'] = 15
test_df['Exterior1st'] = test_df['Exterior1st'].astype('int')

test_df.loc[test_df['Exterior2nd'] == 'CBlock', 'Exterior2nd'] = 1
test_df.loc[test_df['Exterior2nd'] == 'AsbShng', 'Exterior2nd'] = 2
test_df.loc[test_df['Exterior2nd'] == 'Brk Cmn', 'Exterior2nd'] = 3
test_df.loc[test_df['Exterior2nd'] == 'AsphShn', 'Exterior2nd'] = 4
test_df.loc[test_df['Exterior2nd'] == 'Wd Sdng', 'Exterior2nd'] = 5
test_df.loc[test_df['Exterior2nd'] == 'MetalSd', 'Exterior2nd'] = 6
test_df.loc[test_df['Exterior2nd'] == 'Stucco', 'Exterior2nd'] = 7
test_df.loc[test_df['Exterior2nd'] == 'Stone', 'Exterior2nd'] = 8
test_df.loc[test_df['Exterior2nd'] == 'Wd Shng', 'Exterior2nd'] = 9
test_df.loc[test_df['Exterior2nd'] == 'Plywood', 'Exterior2nd'] = 10
test_df.loc[test_df['Exterior2nd'] == 'HdBoard', 'Exterior2nd'] = 11
test_df.loc[test_df['Exterior2nd'] == 'BrkFace', 'Exterior2nd'] = 12
test_df['Exterior2nd'].replace(np.NaN, 13, inplace = True)
test_df.loc[test_df['Exterior2nd'] == 'VinylSd', 'Exterior2nd'] = 13
test_df.loc[test_df['Exterior2nd'] == 'CmentBd', 'Exterior2nd'] = 14
test_df.loc[test_df['Exterior2nd'] == 'ImStucc', 'Exterior2nd'] = 15
test_df.loc[test_df['Exterior2nd'] == 'Other', 'Exterior2nd'] = 16
test_df['Exterior2nd'] = test_df['Exterior2nd'].astype('int')

test_df.loc[test_df['MasVnrType'] == 'BrkCmn', 'MasVnrType'] = 1
test_df['MasVnrType'].replace(np.NaN, 2, inplace = True)
test_df.loc[test_df['MasVnrType'] == 'None', 'MasVnrType'] = 2
test_df.loc[test_df['MasVnrType'] == 'BrkFace', 'MasVnrType'] = 3
test_df.loc[test_df['MasVnrType'] == 'CBlock', 'MasVnrType'] = 4
test_df.loc[test_df['MasVnrType'] == 'Stone', 'MasVnrType'] = 5
test_df['MasVnrType'] = test_df['MasVnrType'].astype('int')

test_df.loc[test_df['ExterQual'] == 'Fa', 'ExterQual'] = 1
test_df.loc[test_df['ExterQual'] == 'TA', 'ExterQual'] = 2
test_df.loc[test_df['ExterQual'] == 'Gd', 'ExterQual'] = 3
test_df.loc[test_df['ExterQual'] == 'Ex', 'ExterQual'] = 4
test_df['ExterQual'] = test_df['ExterQual'].astype('int')

test_df.loc[test_df['ExterCond'] == 'Po', 'ExterCond'] = 1
test_df.loc[test_df['ExterCond'] == 'Fa', 'ExterCond'] = 2
test_df.loc[test_df['ExterCond'] == 'TA', 'ExterCond'] = 3
test_df.loc[test_df['ExterCond'] == 'Gd', 'ExterCond'] = 4
test_df.loc[test_df['ExterCond'] == 'Ex', 'ExterCond'] = 5
test_df['ExterCond'] = test_df['ExterCond'].astype('int')

test_df.loc[test_df['Foundation'] == 'Slab', 'Foundation'] = 1
test_df.loc[test_df['Foundation'] == 'BrkTil', 'Foundation'] = 2
test_df.loc[test_df['Foundation'] == 'Stone', 'Foundation'] = 4
test_df.loc[test_df['Foundation'] == 'Wood', 'Foundation'] = 5
test_df.loc[test_df['Foundation'] == 'CBlock', 'Foundation'] = 3
test_df.loc[test_df['Foundation'] == 'PConc', 'Foundation'] = 6
test_df['Foundation'] = test_df['Foundation'].astype('int')

test_df['BsmtQual'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['BsmtQual'] == 'Fa', 'BsmtQual'] = 1
test_df.loc[test_df['BsmtQual'] == 'TA', 'BsmtQual'] = 2
test_df.loc[test_df['BsmtQual'] == 'Gd', 'BsmtQual'] = 3
test_df.loc[test_df['BsmtQual'] == 'Ex', 'BsmtQual'] = 4
test_df['BsmtQual'] = test_df['BsmtQual'].astype('int')

test_df['BsmtCond'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['BsmtCond'] == 'Po', 'BsmtCond'] = 1
test_df.loc[test_df['BsmtCond'] == 'Fa', 'BsmtCond'] = 2
test_df.loc[test_df['BsmtCond'] == 'TA', 'BsmtCond'] = 3
test_df.loc[test_df['BsmtCond'] == 'Gd', 'BsmtCond'] = 4
test_df['BsmtCond'] = test_df['BsmtCond'].astype('int')

test_df['BsmtExposure'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['BsmtExposure'] == 'No', 'BsmtExposure'] = 1
test_df.loc[test_df['BsmtExposure'] == 'Mn', 'BsmtExposure'] = 2
test_df.loc[test_df['BsmtExposure'] == 'Av', 'BsmtExposure'] = 3
test_df.loc[test_df['BsmtExposure'] == 'Gd', 'BsmtExposure'] = 4
test_df['BsmtExposure'] = test_df['BsmtExposure'].astype('int')

test_df['BsmtFinType1'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['BsmtFinType1'] == 'Unf', 'BsmtFinType1'] = 1
test_df.loc[test_df['BsmtFinType1'] == 'LwQ', 'BsmtFinType1'] = 2
test_df.loc[test_df['BsmtFinType1'] == 'Rec', 'BsmtFinType1'] = 3
test_df.loc[test_df['BsmtFinType1'] == 'BLQ', 'BsmtFinType1'] = 4
test_df.loc[test_df['BsmtFinType1'] == 'ALQ', 'BsmtFinType1'] = 5
test_df.loc[test_df['BsmtFinType1'] == 'GLQ', 'BsmtFinType1'] = 6
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].astype('int')

test_df['BsmtFinType2'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['BsmtFinType2'] == 'Unf', 'BsmtFinType2'] = 1
test_df.loc[test_df['BsmtFinType2'] == 'LwQ', 'BsmtFinType2'] = 2
test_df.loc[test_df['BsmtFinType2'] == 'Rec', 'BsmtFinType2'] = 3
test_df.loc[test_df['BsmtFinType2'] == 'BLQ', 'BsmtFinType2'] = 4
test_df.loc[test_df['BsmtFinType2'] == 'ALQ', 'BsmtFinType2'] = 5
test_df.loc[test_df['BsmtFinType2'] == 'GLQ', 'BsmtFinType2'] = 6
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].astype('int')

test_df.loc[test_df['Heating'] == 'Floor', 'Heating'] = 1
test_df.loc[test_df['Heating'] == 'Grav', 'Heating'] = 2
test_df.loc[test_df['Heating'] == 'Wall', 'Heating'] = 3
test_df.loc[test_df['Heating'] == 'OthW', 'Heating'] = 4
test_df.loc[test_df['Heating'] == 'GasW', 'Heating'] = 5
test_df.loc[test_df['Heating'] == 'GasA', 'Heating'] = 6
test_df['Heating'] = test_df['Heating'].astype('int')

test_df.loc[test_df['HeatingQC'] == 'Po', 'HeatingQC'] = 1
test_df.loc[test_df['HeatingQC'] == 'Fa', 'HeatingQC'] = 2
test_df.loc[test_df['HeatingQC'] == 'TA', 'HeatingQC'] = 3
test_df.loc[test_df['HeatingQC'] == 'Gd', 'HeatingQC'] = 4
test_df.loc[test_df['HeatingQC'] == 'Ex', 'HeatingQC'] = 5
test_df['HeatingQC'] = test_df['HeatingQC'].astype('int')

test_df.loc[test_df['CentralAir'] == 'N', 'CentralAir'] = 0
test_df.loc[test_df['CentralAir'] == 'Y', 'CentralAir'] = 1
test_df['CentralAir'] = test_df['CentralAir'].astype('int')

test_df.loc[test_df['Electrical'] == 'Mix', 'Electrical'] = 1
test_df.loc[test_df['Electrical'] == 'FuseP', 'Electrical'] = 2
test_df.loc[test_df['Electrical'] == 'FuseF', 'Electrical'] = 3
test_df.loc[test_df['Electrical'] == 'FuseA', 'Electrical'] = 4
test_df.loc[test_df['Electrical'] == 'SBrkr', 'Electrical'] = 5
test_df['Electrical'].replace(np.NaN, 5, inplace = True)
test_df['Electrical'] = test_df['Electrical'].astype('int')

test_df.loc[test_df['KitchenQual'] == 'Fa', 'KitchenQual'] = 1
test_df.loc[test_df['KitchenQual'] == 'TA', 'KitchenQual'] = 2
test_df['KitchenQual'].replace(np.NaN, 3, inplace = True)
test_df.loc[test_df['KitchenQual'] == 'Gd', 'KitchenQual'] = 3
test_df.loc[test_df['KitchenQual'] == 'Ex', 'KitchenQual'] = 4
test_df['KitchenQual'] = test_df['KitchenQual'].astype('int')

test_df.loc[test_df['Functional'] == 'Maj2', 'Functional'] = 1
test_df.loc[test_df['Functional'] == 'Sev', 'Functional'] = 2
test_df.loc[test_df['Functional'] == 'Min2', 'Functional'] = 3
test_df.loc[test_df['Functional'] == 'Min1', 'Functional'] = 4
test_df.loc[test_df['Functional'] == 'Maj1', 'Functional'] = 5
test_df.loc[test_df['Functional'] == 'Mod', 'Functional'] = 6
test_df['Functional'].replace(np.NaN, 7, inplace = True)
test_df.loc[test_df['Functional'] == 'Typ', 'Functional'] = 7
test_df['Functional'] = test_df['Functional'].astype('int')

test_df['FireplaceQu'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['FireplaceQu'] == 'Po', 'FireplaceQu'] = 1
test_df.loc[test_df['FireplaceQu'] == 'Fa', 'FireplaceQu'] = 2
test_df.loc[test_df['FireplaceQu'] == 'TA', 'FireplaceQu'] = 3
test_df.loc[test_df['FireplaceQu'] == 'Gd', 'FireplaceQu'] = 4
test_df.loc[test_df['FireplaceQu'] == 'Ex', 'FireplaceQu'] = 5
test_df['FireplaceQu'] = test_df['FireplaceQu'].astype('int')

test_df['GarageType'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['GarageType'] == 'CarPort', 'GarageType'] = 1
test_df.loc[test_df['GarageType'] == 'Detchd', 'GarageType'] = 2
test_df.loc[test_df['GarageType'] == '2Types', 'GarageType'] = 3
test_df.loc[test_df['GarageType'] == 'Basment', 'GarageType'] = 4
test_df.loc[test_df['GarageType'] == 'Attchd', 'GarageType'] = 5
test_df.loc[test_df['GarageType'] == 'BuiltIn', 'GarageType'] = 6
test_df['GarageType'] = test_df['GarageType'].astype('int')

test_df['GarageFinish'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['GarageFinish'] == 'Unf', 'GarageFinish'] = 1
test_df.loc[test_df['GarageFinish'] == 'RFn', 'GarageFinish'] = 2
test_df.loc[test_df['GarageFinish'] == 'Fin', 'GarageFinish'] = 3
test_df['GarageFinish'] = test_df['GarageFinish'].astype('int')

test_df['GarageQual'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['GarageQual'] == 'Po', 'GarageQual'] = 1
test_df.loc[test_df['GarageQual'] == 'Fa', 'GarageQual'] = 2
test_df.loc[test_df['GarageQual'] == 'TA', 'GarageQual'] = 3
test_df.loc[test_df['GarageQual'] == 'Gd', 'GarageQual'] = 4
test_df.loc[test_df['GarageQual'] == 'Ex', 'GarageQual'] = 5
test_df['GarageQual'] = test_df['GarageQual'].astype('int')

test_df['GarageCond'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['GarageCond'] == 'Po', 'GarageCond'] = 1
test_df.loc[test_df['GarageCond'] == 'Fa', 'GarageCond'] = 2
test_df.loc[test_df['GarageCond'] == 'TA', 'GarageCond'] = 3
test_df.loc[test_df['GarageCond'] == 'Gd', 'GarageCond'] = 4
test_df.loc[test_df['GarageCond'] == 'Ex', 'GarageCond'] = 5
test_df['GarageCond'] = test_df['GarageCond'].astype('int')

test_df.loc[test_df['PavedDrive'] == 'N', 'PavedDrive'] = 0
test_df.loc[test_df['PavedDrive'] == 'P', 'PavedDrive'] = 1
test_df.loc[test_df['PavedDrive'] == 'Y', 'PavedDrive'] = 2
test_df['PavedDrive'] = test_df['PavedDrive'].astype('int')

test_df.loc[test_df['PoolArea'] == 0, 'PoolArea'] = 0
test_df.loc[test_df['PoolArea'] != 0, 'PoolArea'] = 1
test_df['PoolArea'] = test_df['PoolArea'].astype('int')

test_df['Fence'].replace(np.NaN, 0, inplace = True)
test_df.loc[test_df['Fence'] == 'MnWw', 'Fence'] = 1
test_df.loc[test_df['Fence'] == 'GdWo', 'Fence'] = 2
test_df.loc[test_df['Fence'] == 'MnPrv', 'Fence'] = 3
test_df.loc[test_df['Fence'] == 'GdPrv', 'Fence'] = 4
test_df['Fence'] = test_df['Fence'].astype('int')

test_df.loc[test_df['MiscFeature'] == 'Othr', 'MiscFeature'] = 1
test_df.loc[test_df['MiscFeature'] == 'Shed', 'MiscFeature'] = 2
test_df.loc[test_df['MiscFeature'] == 'Gar2', 'MiscFeature'] = 3
test_df['MiscFeature'].replace(np.NaN, 4, inplace = True)
test_df.loc[test_df['MiscFeature'] == 'TenC', 'MiscFeature'] = 5
test_df['MiscFeature'] = test_df['MiscFeature'].astype('int')

test_df.loc[test_df['SaleType'] == 'Oth', 'SaleType'] = 1
test_df.loc[test_df['SaleType'] == 'ConLD', 'SaleType'] = 2
test_df.loc[test_df['SaleType'] == 'ConLw', 'SaleType'] = 3
test_df.loc[test_df['SaleType'] == 'COD', 'SaleType'] = 4
test_df['SaleType'].replace(np.NaN, 5, inplace = True)
test_df.loc[test_df['SaleType'] == 'WD', 'SaleType'] = 5
test_df.loc[test_df['SaleType'] == 'ConLI', 'SaleType'] = 6
test_df.loc[test_df['SaleType'] == 'CWD', 'SaleType'] = 7
test_df.loc[test_df['SaleType'] == 'Con', 'SaleType'] = 8
test_df.loc[test_df['SaleType'] == 'New', 'SaleType'] = 9
test_df['SaleType'] = test_df['SaleType'].astype('int')

test_df.loc[test_df['SaleCondition'] == 'AdjLand', 'SaleCondition'] = 1
test_df.loc[test_df['SaleCondition'] == 'Abnorml', 'SaleCondition'] = 2
test_df.loc[test_df['SaleCondition'] == 'Family', 'SaleCondition'] = 3
test_df.loc[test_df['SaleCondition'] == 'Alloca', 'SaleCondition'] = 4
test_df.loc[test_df['SaleCondition'] == 'Normal', 'SaleCondition'] = 5
test_df.loc[test_df['SaleCondition'] == 'Partial', 'SaleCondition'] = 6
test_df['SaleCondition'] = test_df['SaleCondition'].astype('int')

test_df.loc[test_df['MSSubClass'] == 30, 'MSSubClass'] = 1
test_df.loc[test_df['MSSubClass'] == 180, 'MSSubClass'] = 2
test_df.loc[test_df['MSSubClass'] == 45, 'MSSubClass'] = 3
test_df.loc[test_df['MSSubClass'] == 190, 'MSSubClass'] = 4
test_df.loc[test_df['MSSubClass'] == 90, 'MSSubClass'] = 5
test_df.loc[test_df['MSSubClass'] == 160, 'MSSubClass'] = 6
test_df.loc[test_df['MSSubClass'] == 50, 'MSSubClass'] = 7
test_df.loc[test_df['MSSubClass'] == 85, 'MSSubClass'] = 8
test_df.loc[test_df['MSSubClass'] == 40, 'MSSubClass'] = 9
test_df.loc[test_df['MSSubClass'] == 70, 'MSSubClass'] = 10
test_df.loc[test_df['MSSubClass'] == 80, 'MSSubClass'] = 11
test_df.loc[test_df['MSSubClass'] == 20, 'MSSubClass'] = 12
test_df.loc[test_df['MSSubClass'] == 150, 'MSSubClass'] = 12
test_df.loc[test_df['MSSubClass'] == 75, 'MSSubClass'] = 13
test_df.loc[test_df['MSSubClass'] == 120, 'MSSubClass'] = 14
test_df.loc[test_df['MSSubClass'] == 150, 'MSSubClass'] = 15
test_df.loc[test_df['MSSubClass'] == 60, 'MSSubClass'] = 16
test_df['MSSubClass'] = test_df['MSSubClass'].astype('int')

test_df['GarageCars'].replace(np.NaN, 0, inplace = True)

test_df['GarageArea'].replace(np.NaN, 0, inplace = True)

test_df['GarageYrBlt'].replace(np.NaN, garageYearBuiltMean, inplace = True)

test_df['MasVnrArea'].replace(np.NaN, MasVnrAreaMean, inplace = True)

test_df['LotFrontage'].replace(np.NaN, lotFrontageMean, inplace = True)

test_df['BsmtFullBath'].replace(np.NaN, 0, inplace = True)

test_df['BsmtHalfBath'].replace(np.NaN, 0, inplace = True)

test_df['BsmtFinSF1'].replace(np.NaN, 0, inplace = True)

test_df['BsmtFinSF2'].replace(np.NaN, 0, inplace = True)

test_df['BsmtUnfSF'].replace(np.NaN, 0, inplace = True)

test_df['TotalBsmtSF'].replace(np.NaN, 0, inplace = True)

# Feature Selection

arr_test = test_df.to_numpy()

test_OverallQuality = np.array(arr_test[:,17], np.int8)
test_size = test_OverallQuality.size

test_GroundLivingArea = np.array(arr_test[:,46], np.float32)
test_GroundLivingArea = test_GroundLivingArea/100

test_Neighborhood = np.array(arr_test[:,12], np.int8)

test_ExteriorQuality = np.array(arr_test[:,27], np.int8)

test_KitchenQuality = np.array(arr_test[:,53], np.int8)

test_GarageCars = np.array(arr_test[:,61], np.int8)

test_GarageArea = np.array(arr_test[:,62], np.float32)
test_GarageArea = test_GarageArea/100

test_BasementQuality = np.array(arr_test[:,30], np.int8)

test_TotalBasementSF = np.array(arr_test[:,38], np.float32)
test_TotalBasementSF = test_TotalBasementSF/100

test_1stFlrSF = np.array(arr_test[:,43], np.float32)
test_1stFlrSF = test_1stFlrSF/100

test_FullBath = np.array(arr_test[:,49], np.int8)

test_GarageFinish = np.array(arr_test[:,60], np.int8)

test_TotalRoomsAboveGround = np.array(arr_test[:,54], np.int8)

test_FireplaceQuality = np.array(arr_test[:,57], np.int8)

test_YearRemodeled = np.array(arr_test[:,20], np.float32)
test_YearRemodeled = test_YearRemodeled/100

test_Foundation = np.array(arr_test[:,29], np.int8)

test_GarageType = np.array(arr_test[:,58], np.int8)

test_GarageYearBuilt = np.array(arr_test[:,59], np.float32)
test_GarageYearBuilt = test_GarageYearBuilt/100

test_MSSubClass = np.array(arr_test[:,1], np.int8)

test_MasVnrArea = np.array(arr_test[:,26], np.float32)
test_MasVnrArea = test_MasVnrArea/100

test_Fireplaces = np.array(arr_test[:,56], np.int8)

test_HeatingQuality = np.array(arr_test[:,40], np.int8)

test_MasVnrType = np.array(arr_test[:,25], np.int8)

test_BasementFinSF1 = np.array(arr_test[:,34], np.float32)
test_BasementFinSF1 = test_BasementFinSF1/100

test_BasementExposure = np.array(arr_test[:,32], np.int8)

test_ExteriorFirst = np.array(arr_test[:,23], np.int8)

test_SaleType = np.array(arr_test[:,78], np.int8)

test_ExteriorSecond = np.array(arr_test[:,24], np.int8)

test_LotFrontage = np.array(arr_test[:,23], np.float32)
test_LotFrontage = test_LotFrontage/10

test_MSZoning = np.array(arr_test[:,2], np.int8)

test_WoodDeckSF = np.array(arr_test[:,66], np.float32)
test_WoodDeckSF = test_WoodDeckSF/100

test_2ndFlrSF = np.array(arr_test[:,44], np.float32)
test_2ndFlrSF = test_2ndFlrSF/100

test_OpenPorchSF = np.array(arr_test[:,67], np.float32)
test_OpenPorchSF = test_OpenPorchSF/100

test_BsmtFinType1 = np.array(arr_test[:,33], np.int8)

test_HalfBath = np.array(arr_test[:,50], np.int8)

test_GarageQual = np.array(arr_test[:,63], np.int8)

test_HouseStyle = np.array(arr_test[:,16], np.int8)

test_LotShape = np.array(arr_test[:,7], np.int8)

test_LotArea = np.array(arr_test[:,4], np.float32)
test_LotArea = test_LotArea/10000

test_GarageCond = np.array(arr_test[:,64], np.int8)

test_CentralAir = np.array(arr_test[:,41], np.int8)

test_RoofStyle = np.array(arr_test[:,21], np.int8)

test_SaleCondition = np.array(arr_test[:,79], np.int8)

test_Electrical = np.array(arr_test[:,42], np.int8)

test_PavedDrive = np.array(arr_test[:,65], np.int8)

test_BsmtFullBath = np.array(arr_test[:,47], np.int8)

test_BsmtUnfSF = np.array(arr_test[:,37], np.float32)
test_BsmtUnfSF = test_BsmtUnfSF/100

test_BsmtCond = np.array(arr_test[:,31], np.int8)

test_BldgType = np.array(arr_test[:,15], np.int8)

test_Condition1 = np.array(arr_test[:,13], np.int8)

test_BedroomAbvGr = np.array(arr_test[:,51], np.int8)

test_RoofMatl = np.array(arr_test[:,22], np.int8)

test_LandContour = np.array(arr_test[:,8], np.int8)

test_Alley = np.array(arr_test[:,6], np.int8)

test_Functional = np.array(arr_test[:,5], np.int8)

test_LotConfig = np.array(arr_test[:,10], np.int8)

test_Heating = np.array(arr_test[:,39], np.int8)

test_ScreenPorch = np.array(arr_test[:,70], np.float32)
test_ScreenPorch = test_ScreenPorch/100

test_Condition2 = np.array(arr_test[:,14], np.int8)

test_PoolArea = np.array(arr_test[:,71], np.int8)

test_MiscFeature = np.array(arr_test[:,74], np.int8)

test_LandSlope = np.array(arr_test[:,11], np.int8)

test_OverallCond = np.array(arr_test[:,18], np.int8)

test_EnclosedPorch = np.array(arr_test[:,68], np.float32)
test_EnclosedPorch = test_EnclosedPorch/100

test_KitchenAbvGr = np.array(arr_test[:,52], np.int8)

test_Fence = np.array(arr_test[:,73], np.int8)

test_X = np.ones(test_size, np.int8)

index = np.array(arr_test[:,0], np.int64)

predictions = np.zeros(test_size, np.float64)
for i in range(0, test_size):
    predictions[i] = test_X[i]*theta[0] + test_OverallQuality[i]*theta[1] + test_GroundLivingArea[i]*theta[2] + test_Neighborhood[i]*theta[3] + test_ExteriorQuality[i]*theta[4] + test_KitchenQuality[i]*theta[5] + test_GarageCars[i]*theta[6] + test_GarageArea[i]*theta[7] + test_BasementQuality[i]*theta[8] + test_TotalBasementSF[i]*theta[9] + test_1stFlrSF[i]*theta[10] + test_FullBath[i]*theta[11] + test_GarageFinish[i]*theta[12] + test_TotalRoomsAboveGround[i]*theta[13] + test_FireplaceQuality[i]*theta[14] + test_YearRemodeled[i]*theta[15] + test_Foundation[i]*theta[16] + test_GarageType[i]*theta[17] + test_GarageYearBuilt[i]*theta[18] + test_MSSubClass[i]*theta[19] + test_MasVnrArea[i]*theta[20] + test_Fireplaces[i]*theta[21] + test_HeatingQuality[i]*theta[22] + test_MasVnrType[i]*theta[23] + test_BasementFinSF1[i]*theta[24] + test_BasementExposure[i]*theta[25] + test_ExteriorFirst[i]*theta[26] + test_SaleType[i]*theta[27] + test_ExteriorSecond[i]*theta[28] + test_LotFrontage[i]*theta[29] + test_MSZoning[i]*theta[30] + test_WoodDeckSF[i]*theta[31] + test_2ndFlrSF[i]*theta[32] + test_OpenPorchSF[i]*theta[33] + test_BsmtFinType1[i]*theta[34] + test_HalfBath[i]*theta[35] + test_GarageQual[i]*theta[36] + test_HouseStyle[i]*theta[37] + test_LotShape[i]*theta[38] + test_LotArea[i]*theta[39] + test_GarageCond[i]*theta[40] + test_CentralAir[i]*theta[41] + test_RoofStyle[i]*theta[42] + test_SaleCondition[i]*theta[43] + test_Electrical[i]*theta[44] + test_PavedDrive[i]*theta[45] + test_BsmtFullBath[i]*theta[46] + test_BsmtUnfSF[i]*theta[47] + test_BsmtCond[i]*theta[48] + test_BldgType[i]*theta[49] + test_Condition1[i]*theta[50] + test_BedroomAbvGr[i]*theta[51] + test_RoofMatl[i]*theta[52] + test_LandContour[i]*theta[53] + test_Alley[i]*theta[54] + test_Functional[i]*theta[55] + test_LotConfig[i]*theta[56] + test_Heating[i]*theta[57] + test_ScreenPorch[i]*theta[58] + test_Condition2[i]*theta[59] + test_PoolArea[i]*theta[60] + test_MiscFeature[i]*theta[61] + test_LandSlope[i]*theta[62] + test_OverallCond[i]*theta[63] + test_EnclosedPorch[i]*theta[64] + test_KitchenAbvGr[i]*theta[65] + test_Fence[i]*theta[66]                                                                                                                                            
        
f = open("theta.txt", "w")    
for i in range(0,67):
    f.write(str(theta[i]) + "\n")
f.close

df_predictions = pd.DataFrame()
df_predictions['Id'] = index
df_predictions['SalePrice'] = predictions
df_predictions.to_csv('Predictions.csv', index = False)        
        
        
        
        
        
        

