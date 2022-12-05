import csv
import numpy as np

def extractHourlyLoad(loadURL):
    loadData = csv.reader(open(loadURL, 'r'), delimiter=',')
    hourlyLoad = []
    for hour in loadData:  # loop through csv file and place kwh load values in an array
        hourlyLoad.append(float(hour[2]))
    return hourlyLoad


def calcMonthlyLoad(hourlyLoad, daysInMonth):
    monthlyLoad = []
    hourCount = 0  # keeps track of the hour of the year

    for month in range(12):
        loadSum = 0  # load in a month

        for hour in range(daysInMonth[month] * 24):  # loops through hours in a month and takes the sum of the kwh loads
            loadSum += hourlyLoad[hourCount]
            hourCount += 1

        monthlyLoad.append(loadSum)

    return monthlyLoad


def calcChargeForFlatRate(Cbuy, load):
    yearlyUsageCharge = []  # usage charge through the year

    for month in range(12):  # cycles through 12 months
        monthlyUsageCharge = Cbuy[month] * load[month]  # usage charge for month
        yearlyUsageCharge.append(monthlyUsageCharge)  # adds month's usage charge to total

    return yearlyUsageCharge


def calcChargePerTier(m, Cbuy, Tmax, load):
    remainingLoad = load[m]  # load that has not been charged yet

    # if the load is larger than the first tier we want to charge the first tier kwh at the first tier rate
    # then we want to save the remainder for the next tier

    if Tmax[m][0] < load[m]:  # checks if the monthly load surpasses the first tier
        usageCharge = Cbuy[m][0] * Tmax[m][0]  # charge for the first tier
        remainingLoad -= Tmax[m][0]  # load left over after one tier
    else:
        usageCharge = Cbuy[m][0] * load[m]  # charge for the first tier
        return usageCharge

    if Tmax[m][1] < load[m]:  # checks if the monthly load surpasses the second tier
        usageCharge += Cbuy[m][1] * (Tmax[m][1] - Tmax[m][0])  # adds charge for the second tier
        remainingLoad -= (Tmax[m][1] - Tmax[m][0])  # load left over after two tiers
    else:
        usageCharge += Cbuy[m][1] * remainingLoad  # adds charge for the second tier using left over load
        return usageCharge

    usageCharge += Cbuy[m][2] * remainingLoad  # adds charge for the third tier using left over load
    return usageCharge


def calcChargeForTieredRate(Cbuy, Tmax, load):
    yearlyUsageCharge = []  # stores the total charge for each month

    for m in range(12):
        monthlyUsageCharge = calcChargePerTier(m, Cbuy, Tmax, load)
        yearlyUsageCharge.append(monthlyUsageCharge)

    return yearlyUsageCharge


def calcTouCbuy(onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, Day, holidays):
    Cbuy = np.zeros(8760)
    for m in range(1, 13):
        t_start = (24 * np.sum(Day[0:m - 1]) + 1).astype(int)
        t_end = (24 * np.sum(Day[0:m])).astype(int)
        t_index = list(range(t_start, t_end + 1))
        t_index = np.array(t_index).astype(int)
        nt = len(t_index)

        if Month[m - 1] == 1:  # for summer
            tp = np.array(onHours[1]).astype(int)
            tm = np.array(midHours[1]).astype(int)
            toff = np.array(offHours[1]).astype(int)
            P_peak = onPrice[1]
            P_mid = midPrice[1]
            P_offpeak = offPrice[1]
        else:  # for winter
            tp = np.array(onHours[0]).astype(int)
            tm = np.array(midHours[0]).astype(int)
            toff = np.array(offHours[0]).astype(int)
            P_peak = onPrice[0]
            P_mid = midPrice[0]
            P_offpeak = offPrice[0]

        print(tp, tm, toff)

        Cbuy[t_index - 1] = P_offpeak  # set all hours to offpeak by default
        for d in range(1, Day[m - 1] + 1):
            idx0 = np.array(t_index[tp] + 24 * (d - 1))
            Cbuy[idx0 - 1] = P_peak
            idx1 = np.array(t_index[tm] + 24 * (d - 1))
            Cbuy[idx1 - 1] = P_mid

    for d in range(1, 365):
        if ((d - 1) % 7) >= 5:
            st = 24 * (d - 1) + 1
            ed = 24 * d
            Cbuy[range(st, ed)] = P_offpeak

    for d in range(1, 365):
        if d in holidays:
            st = 24 * (d - 1) + 1
            ed = 24 * d
            Cbuy[range(st, ed)] = P_offpeak

    return Cbuy


def calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, hourlyLoad, Month, daysInMonth, holidays):
    yearlyUsageCharge = []
    Cbuy = calcTouCbuy(onPrice, midPrice, offPrice, onHours, midHours, offHours, Month, daysInMonth, holidays)
    hourCount = 0

    for m in range(12):
        usageCharge = 0
        for hour in range(daysInMonth[m] * 24):
            usageCharge += hourlyLoad[hourCount] * Cbuy[hourCount]
            hourCount += 1

        yearlyUsageCharge.append(usageCharge)

    return yearlyUsageCharge


def calcAdj(subtotalCharge, percentAdj):  # adds a percentage adjustment, to be calculated after all other charges
    totalCharge = []
    for month in subtotalCharge:
        totalCharge.append(month * (1 + (percentAdj / 100)))

    return totalCharge


def calcFlatRate(price, load):
    Cbuy = np.zeros(12)
    for m in range(12):
        Cbuy[m] = price

    flatCharge = calcChargeForFlatRate(Cbuy, load)

    return flatCharge


def calcSeasonalRate(prices, load, months):
    Cbuy = np.zeros(12)
    for m in range(12):
        if months[m] == 0:
            Cbuy[m] = prices[0]
        else:
            Cbuy[m] = prices[1]

    seasonalCharge = calcChargeForFlatRate(Cbuy, load)

    return seasonalCharge


def calcMonthlyRate(prices, load):
    monthlyCharge = calcChargeForFlatRate(prices, load)

    return monthlyCharge


def calcTieredRate(prices, tierMax, load):
    Cbuy = np.zeros(12, dtype=np.ndarray)
    Tmax = np.zeros(12, dtype=np.ndarray)
    for m in range(12):
        Cbuy[m] = prices
        Tmax[m] = tierMax

    tieredCharge = calcChargeForTieredRate(Cbuy, Tmax, load)

    return tieredCharge


def calcSeasonalTieredRate(prices, tierMax, load, months):
    Cbuy = np.zeros(12, dtype=np.ndarray)
    Tmax = np.zeros(12, dtype=np.ndarray)
    for m in range(12):
        if months[m] == 0:
            Cbuy[m] = prices[0]
            Tmax[m] = tierMax[0]
        else:
            Cbuy[m] = prices[1]
            Tmax[m] = tierMax[1]

    seasonalTieredCharge = calcChargeForTieredRate(Cbuy, Tmax, load)

    return seasonalTieredCharge


def calcMonthlyTieredRate(prices, tierMax, load):
    monthlyTieredCharge = calcChargeForTieredRate(prices, tierMax, load)

    return monthlyTieredCharge


########################################################################################################################

def getTotalCharge(rateStructure = 7):
    # define which utility rate is used
    # rateStructure = 7
    '''
    1 = flat rate
    2 = seasonal rate
    3 = monthly rate
    4 = tiered rate
    5 = seasonal tiered rate
    6 = monthly tiered rate
    7 = time of use rate
    '''

    # Months
    months = np.zeros(12)

    # days in each month
    daysInMonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # percentage adjustment
    percentAdj = 2

    # name of the file with the hourly load
    loadURL = "Load.csv"

    # load for each hour of year
    hourlyLoad = extractHourlyLoad(loadURL)

    # total monthly charge
    if rateStructure == 1:  # flat rate
        # price for flat rate
        flatPrice = 0.112

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcFlatRate(flatPrice, monthlyLoad)

    elif rateStructure == 2:  # seasonal rate
        # prices for seasonal rate [winter, summer]
        seasonalPrices = [0.13, 0.17]
        # define summer season
        months[4:11] = 1

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcSeasonalRate(seasonalPrices, monthlyLoad, months)

    elif rateStructure == 3:  # monthly rate
        # prices for monthly rate [Jan-Dec]
        monthlyPrices = [0.15, 0.15, 0.15, 0.15, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.15, 0.15]

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcMonthlyRate(monthlyPrices, monthlyLoad)

    elif rateStructure == 4:  # tiered rate
        # prices and max kwh limits [tier 1, 2, 3]
        tieredPrices = [0.1, 0.12, 0.15]
        tierMax = [500, 1000, 4000]

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcTieredRate(tieredPrices, tierMax, monthlyLoad)

    elif rateStructure == 5:  # seasonal tiered rate
        # prices and max kwh limits [winter, summer][tier 1, 2, 3]
        seasonalTieredPrices = [
            [0.05, 0.08, 0, 14],
            [0.09, 0.13, 0.2]
        ]
        seasonalTierMax = [
            [400, 800, 4000],
            [1000, 1500, 4000]
        ]
        # define summer season
        months[4:11] = 1

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcSeasonalTieredRate(seasonalTieredPrices, seasonalTierMax, monthlyLoad, months)

    elif rateStructure == 6:  # monthly tiered rate
        # prices and max kwh limits [Jan-Dec][tier 1, 2, 3]
        monthlyTieredPrices = [
            [0.08, 0.10, 0.14],
            [0.08, 0.10, 0.14],
            [0.08, 0.10, 0.14],
            [0.08, 0.10, 0.14],
            [0.12, 0.14, 0.20],
            [0.12, 0.14, 0.20],
            [0.12, 0.14, 0.20],
            [0.12, 0.14, 0.20],
            [0.12, 0.14, 0.20],
            [0.12, 0.14, 0.20],
            [0.08, 0.10, 0.14],
            [0.08, 0.10, 0.14]
        ]
        monthlyTierLimits = [
            [600, 1500, 4000],
            [600, 1500, 4000],
            [600, 1500, 4000],
            [600, 1500, 4000],
            [800, 1500, 4000],
            [800, 1500, 4000],
            [800, 1500, 4000],
            [800, 1500, 4000],
            [800, 1500, 4000],
            [800, 1500, 4000],
            [600, 1500, 4000],
            [600, 1500, 4000]
        ]

        monthlyLoad = calcMonthlyLoad(hourlyLoad, daysInMonth)
        subtotalCharge = calcMonthlyTieredRate(monthlyTieredPrices, monthlyTierLimits, monthlyLoad)

    elif rateStructure == 7:  # time of use rate
        # prices and time of use hours [winter, summer]
        onPrice = [0.1516, 0.3215]
        midPrice = [0, 0.1827]
        offPrice = [0.1098, 0.1323]
        onHours = [
            [17, 18, 19],
            [17, 18, 19]
        ]
        midHours = [
            [],
            [12, 13, 14, 15, 16, 20, 21, 22, 23]
        ]
        offHours = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        ]
        # define summer season
        months[5:9] = 1
        # Holidays definition based on the number of the day in 365 days format
        holidays = [10, 50, 76, 167, 298, 340]

        subtotalCharge = calcTouRate(onPrice, midPrice, offPrice, onHours, midHours, offHours, hourlyLoad, months, daysInMonth, holidays)


    totalCharge = calcAdj(subtotalCharge, percentAdj)
    return totalCharge
