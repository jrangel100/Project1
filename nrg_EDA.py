# Import Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import requests
import time
from scipy.stats import linregress


############

# Import Production Data (and State Index)
# Production Data Source
# https://www.kaggle.com/datasets/kevinmorgado/us-energy-generation-2001-2022?select=organised_Gen.csv

# Load the State Index CSV file
state_df = pd.read_csv("Resources/states.csv", usecols = ['State', 'Code'])

# Load the Production CSV file and exclude first column
col_list = ['YEAR', 'MONTH', 'STATE', 'TYPE OF PRODUCER', 'ENERGY SOURCE', 'GENERATION (Megawatthours)']
prod_df = pd.read_csv("Resources/organised_Gen.csv", usecols = col_list)

# Assign title case to column names and rename 'State'
col_list_cased = [col.title() for col in col_list]
prod_df.columns = col_list_cased

# Filter for years 2002 - 2021 to match investment data set
prod_df = prod_df.loc[(prod_df['Year'] > 2001) & (prod_df['Year'] < 2022) & (prod_df['State'] != 'US-TOTAL') & (prod_df['Type Of Producer'] == "Total Electric Power Industry")]

# Filter for common energy types across both data sets
prod_source_list = ['Geothermal', 'Hydroelectric Conventional', 'Other Biomass', 'Solar Thermal and Photovoltaic', 'Wind']
prod_df = prod_df.loc[prod_df['Energy Source'].isin(prod_source_list)]

# Rename energy sources to common convention
prod_df = prod_df.replace(['Hydroelectric Conventional', 'Other Biomass', 'Solar Thermal and Photovoltaic'], 
                          ['Hydroelectric', 'Biomass', 'Solar'])

# Display sample data
prod_df.head()

####################

# Import Investment Data
# Investment Data Source
# https://www.wctsservices.usda.gov/Energy/Downloads

# Load the Investment Excel file, "Detailed" Sheet
inv_df = pd.read_excel("Resources/EnergyInvestments_DataDownloads.xlsx", sheet_name = "Detailed")

# Filter for common energy types across both data sets
inv_source_list = ['Geothermal', 'Hydroelectric', 'Renewable Biomass', 'Solar', 'Wind']
inv_df = inv_df.loc[inv_df['Energy Type'].isin(inv_source_list)]

# Rename energy sources to common convention
inv_df = inv_df.replace('Renewable Biomass', 'Biomass')

# Join with State Index data to get State Abbreviation column
inv_df = pd.merge(inv_df, state_df, on = "State")

# Rename columns
inv_df.rename(columns={"State": "State Name", "Code": "State", "Energy Type": "Energy Source", "Program_Name": "Program Name"}, inplace = True)

# Display sample data
inv_df.head()

##########################

## Look at the trend of total production ('Generation (Megawatthours)') from 2002 - 2021 (all states combined)
# Groupby 'Year'
prod_df_year = prod_df.groupby(["Year"])

# x-axis
x_axis = prod_df_year['Year'].first()

# y-axis
y_axis = prod_df_year['Generation (Megawatthours)'].sum()

# Line plot
plt.figure(figsize = (12, 4))
plt.plot(x_axis, y_axis)
plt.xlabel("Year")
plt.xticks(np.arange(x_axis.min(), x_axis.max() + 1, 1.0))
plt.ylabel("Generation (Megawatthours)")
plt.show()

###################


## Look at the trend of total production ('Generation (Megawatthours)') from 2002 - 2021 per Energy Source (all states combined)
# Energy Source List
source_list = ['Biomass', 'Geothermal', 'Hydroelectric', 'Solar', 'Wind']

# Color List
colors = ['red', 'green', 'blue', 'orange', 'lightblue']

# y-axis - Filter for 'Energy Source'
prod_df_bio = prod_df.loc[prod_df['Energy Source'] == source_list[0]].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_geo = prod_df.loc[prod_df['Energy Source'] == source_list[1]].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_hyd = prod_df.loc[prod_df['Energy Source'] == source_list[2]].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_sol = prod_df.loc[prod_df['Energy Source'] == source_list[3]].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_wnd = prod_df.loc[prod_df['Energy Source'] == source_list[4]].groupby(["Year"])['Generation (Megawatthours)'].sum()

# Line plot
plt.figure(figsize = (12, 4))
bio, = plt.plot(x_axis, prod_df_bio, color = colors[0], label = source_list[0])
geo, = plt.plot(x_axis, prod_df_geo, color = colors[1], label = source_list[1])
hyd, = plt.plot(x_axis, prod_df_hyd, color = colors[2], label = source_list[2])
sol, = plt.plot(x_axis, prod_df_sol, color = colors[3], label = source_list[3])
wnd, = plt.plot(x_axis, prod_df_wnd, color = colors[4], label = source_list[4])
plt.xlabel("Year")
plt.xticks(np.arange(x_axis.min(), x_axis.max() + 1, 1.0))
plt.ylabel("Generation (Megawatthours)")
plt.legend(handles = [bio, geo, hyd, sol, wnd], loc = "best")
plt.show()


######################


## Look at the trend of total production ('Generation (Megawatthours)') from 2002 - 2021 per Energy Source (all states combined)

# Filter for 2002, 2021 and ENergy Source
prod_df_bio_02 = prod_df.loc[(prod_df['Energy Source'] == source_list[0]) & (prod_df['Year'] == 2002)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_bio_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[0]) & (prod_df['Year'] == 2021)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_geo_02 = prod_df.loc[(prod_df['Energy Source'] == source_list[1]) & (prod_df['Year'] == 2002)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_geo_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[1]) & (prod_df['Year'] == 2021)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_hyd_02 = prod_df.loc[(prod_df['Energy Source'] == source_list[2]) & (prod_df['Year'] == 2002)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_hyd_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[2]) & (prod_df['Year'] == 2021)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_sol_02 = prod_df.loc[(prod_df['Energy Source'] == source_list[3]) & (prod_df['Year'] == 2002)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_sol_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[3]) & (prod_df['Year'] == 2021)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_wnd_02 = prod_df.loc[(prod_df['Energy Source'] == source_list[4]) & (prod_df['Year'] == 2002)].groupby(["Year"])['Generation (Megawatthours)'].sum()
prod_df_wnd_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[4]) & (prod_df['Year'] == 2021)].groupby(["Year"])['Generation (Megawatthours)'].sum()

# Build summary table showing descending movement per energy source
summary = pd.DataFrame({'Energy Source': source_list,
                       'Movement 2021 vs 2002':[(list(prod_df_bio_21)[0] / list(prod_df_bio_02)[0]) - 1,
                                                (list(prod_df_geo_21)[0] / list(prod_df_geo_02)[0]) - 1,
                                                (list(prod_df_hyd_21)[0] / list(prod_df_hyd_02)[0]) - 1,
                                                (list(prod_df_sol_21)[0] / list(prod_df_sol_02)[0]) - 1,
                                                (list(prod_df_wnd_21)[0] / list(prod_df_wnd_02)[0]) - 1]})
summary = summary.sort_values(by=['Movement 2021 vs 2002'], ascending = False)
summary['Movement 2021 vs 2002'] = summary['Movement 2021 vs 2002'].map("{:,.0%}".format)
summary




######################


## Look at the trend of share total production ('Generation (Megawatthours)') every five years from 2002 - 2021 per Energy Source (all states combined)

# Filter for every five years - 2002, 2007, 2012, 2017, 2021
prod_df_02 = prod_df.loc[prod_df['Year'] == 2002].groupby(['Energy Source'])['Generation (Megawatthours)'].sum()
prod_df_07 = prod_df.loc[prod_df['Year'] == 2007].groupby(['Energy Source'])['Generation (Megawatthours)'].sum()
prod_df_12 = prod_df.loc[prod_df['Year'] == 2012].groupby(['Energy Source'])['Generation (Megawatthours)'].sum()
prod_df_17 = prod_df.loc[prod_df['Year'] == 2017].groupby(['Energy Source'])['Generation (Megawatthours)'].sum()
prod_df_21 = prod_df.loc[prod_df['Year'] == 2021].groupby(['Energy Source'])['Generation (Megawatthours)'].sum()

# y labels
label_source = prod_df.groupby(['Energy Source'])['Energy Source'].first()

# Build pie chart subplots
fig, axs = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
axs[0].pie(list(prod_df_02), colors = colors, autopct="%1.1f%%")
axs[0].title.set_text('2002')
axs[1].pie(list(prod_df_07), colors = colors, autopct="%1.1f%%")
axs[1].title.set_text('2007')
axs[2].pie(list(prod_df_12), colors = colors, autopct="%1.1f%%")
axs[2].title.set_text('2012')
axs[3].pie(list(prod_df_17), colors = colors, autopct="%1.1f%%")
axs[3].title.set_text('2017')
axs[4].pie(list(prod_df_21), colors = colors, autopct="%1.1f%%")
axs[4].title.set_text('2021')
fig.suptitle('Share of Total Generation (Megawatthours)')
fig.legend(list(label_source), loc='lower center', ncol=len(list(label_source)), bbox_transform=fig.transFigure)

###########################

## Look at share of total production ('Generation (Megawatthours)') per Energy Source per State in 2021

# Filter by 2021 and group by State for ALL Energy Sources combined
prod_df_state = prod_df.loc[prod_df['Year'] == 2021].groupby(['State'])['Generation (Megawatthours)'].sum()

# Filter by 2021 and group by State for each Energy Source
prod_df_bio_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[0]) & (prod_df['Year'] == 2021)].groupby(["State"])['Generation (Megawatthours)'].sum()
prod_df_geo_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[1]) & (prod_df['Year'] == 2021)].groupby(["State"])['Generation (Megawatthours)'].sum()
prod_df_hyd_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[2]) & (prod_df['Year'] == 2021)].groupby(["State"])['Generation (Megawatthours)'].sum()
prod_df_sol_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[3]) & (prod_df['Year'] == 2021)].groupby(["State"])['Generation (Megawatthours)'].sum()
prod_df_wnd_21 = prod_df.loc[(prod_df['Energy Source'] == source_list[4]) & (prod_df['Year'] == 2021)].groupby(["State"])['Generation (Megawatthours)'].sum()

# Merge datasets to show evergy sources side-by-side
state_pct = pd.merge(prod_df_state, prod_df_bio_21, how = 'left', on = "State")
state_pct.rename(columns={"Generation (Megawatthours)_x": "Total", "Generation (Megawatthours)_y": source_list[0]}, inplace = True)
state_pct = pd.merge(state_pct, prod_df_geo_21, how = 'left', on = "State")
state_pct.rename(columns={"Generation (Megawatthours)": source_list[1]}, inplace = True)
state_pct = pd.merge(state_pct, prod_df_hyd_21, how = 'left', on = "State")
state_pct.rename(columns={"Generation (Megawatthours)": source_list[2]}, inplace = True)
state_pct = pd.merge(state_pct, prod_df_sol_21, how = 'left', on = "State")
state_pct.rename(columns={"Generation (Megawatthours)": source_list[3]}, inplace = True)
state_pct = pd.merge(state_pct, prod_df_wnd_21, how = 'left', on = "State")
state_pct.rename(columns={"Generation (Megawatthours)": source_list[4]}, inplace = True)
state_pct = state_pct.fillna(0)

# Create new columns for % share of total production
state_pct['Biomass %'] = (state_pct[source_list[0]] / state_pct['Total'])
state_pct['Geothermal %'] = (state_pct[source_list[1]] / state_pct['Total'])
state_pct['Hydroelectric %'] = (state_pct[source_list[2]] / state_pct['Total'])
state_pct['Solar %'] = (state_pct[source_list[3]] / state_pct['Total'])
state_pct['Wind %'] = (state_pct[source_list[4]] / state_pct['Total'])
state_pct = state_pct.iloc[:, 6:11]
state_pct



############################


## Look at Top 5 states per Energy Source in 2021

# Sort state_pct table by each Energy Source (descending)
bio_pct = state_pct.sort_values(by=['Biomass %'], ascending = False)
bio = list(bio_pct.iloc[0:5, 0].index)
geo_pct = state_pct.sort_values(by=['Geothermal %'], ascending = False)
geo = list(geo_pct.iloc[0:5, 1].index)
hyd_pct = state_pct.sort_values(by=['Hydroelectric %'], ascending = False)
hyd = list(hyd_pct.iloc[0:5, 2].index)
sol_pct = state_pct.sort_values(by=['Solar %'], ascending = False)
sol = list(sol_pct.iloc[0:5, 3].index)
wnd_pct = state_pct.sort_values(by=['Wind %'], ascending = False)
wnd = list(wnd_pct.iloc[0:5, 4].index)

# Create summary dataframe to show the top 5 states for each Energy Source
summary = pd.DataFrame({'Rank': np.arange(1, 6, 1),
                        'Biomass %': bio,
                        'Geothermal %': geo,
                        'Hydroelectric %': hyd,
                        'Solar %': sol,
                        'Wind %': wnd})
summary


##########################


# Import Census Data
# Census Data Sources

# https://www.census.gov/data/datasets/time-series/demo/popest/2010s-state-total.html
# https://www.census.gov/data/tables/time-series/demo/popest/2020s-state-total.html
# https://www.census.gov/programs-surveys/popest/technical-documentation/research/evaluation-estimates.2010.html#list-tab-Y660N3MTL49GQLLYDJ

years = [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016,
        2017, 2018, 2019, 2020, 2021]

census_df = pd.DataFrame(columns=['State', 'Population', 'Year'])

state = []
population = []
iterationyear = []

for year in years:
        
    if year < 2011:
        
        popappend = "POPESTIMATE" + str(year)
        
        pop_df = pd.read_csv("Resources/co-est2010-alldata.csv", encoding='ISO-8859-1')
        pop_df = pop_df[[popappend, 'STNAME', 'CTYNAME']]
        pop_df = pop_df.rename(columns={popappend: "Population", "STNAME": "State"})
       
        # This specific dataset is broken down by city pop as well, but the full state pop
        # is held in a 'city' that is the state name, so dropping any rows where
        # the state name is not equal to the city name.
        pop_df = pop_df[pop_df.State == pop_df.CTYNAME]
       
        for ind in pop_df.index:
            state.append(pop_df['State'][ind])
            population.append(pop_df['Population'][ind])
            iterationyear.append(year)
            
    if (year > 2010 and year < 2020):
    
        popappend = "POPESTIMATE" + str(year)
        pop_df = pd.read_csv("Resources/nst-est2019-alldata.csv", encoding='ISO-8859-1')
        pop_df = pop_df[[popappend, 'NAME']]
        pop_df = pop_df.rename(columns={popappend: "Population", "NAME": "State"})
        
        for ind in pop_df.index:
            state.append(pop_df['State'][ind])
            population.append(pop_df['Population'][ind])
            iterationyear.append(year)
    
    if year > 2019:
    
        popappend = "POPESTIMATE" + str(year)
        pop_df = pd.read_csv("Resources/NST-EST2021-alldata.csv", encoding='ISO-8859-1')
        pop_df = pop_df[[popappend, 'NAME']]
        pop_df = pop_df.rename(columns={popappend: "Population", "NAME": "State"})
        
        for ind in pop_df.index:
            state.append(pop_df['State'][ind])
            population.append(pop_df['Population'][ind])
            iterationyear.append(year)
        
        
census_df['State'] = state
census_df['Population'] = population
census_df['Year'] = iterationyear
                                     
#print(census_df)

cleaned_census_df = census_df.groupby(['State', 'Year'])["Population", "Year"].sum()

cleaned_census_df = pd.merge(cleaned_census_df, state_df, how = "left", on = "State")

cleaned_census_df.rename(columns={"State": "State Name", "Code": "State"}, inplace = True)
cleaned_census_df

#################################

#Get all Energy sources grouped into States instead of county
invest_df = inv_df.drop(columns=['County','Agency','Program Name','Congressional District','Zip Code','Description'])

#Getting all of the data for each year under each state
invest_df.groupby(['State', 'Year', 'Energy Source'], as_index=False).agg('sum')
invest_df

##############################


#Aggregating the total amount of investment throughout the entire 2 decades by each State and Energy Source

wind_inv = invest_df.loc[invest_df['Energy Source'] == 'Wind'].groupby(['State'])['Total Amount of Assistance'].sum()
bio_inv = invest_df.loc[invest_df['Energy Source'] == 'Biomass'].groupby(['State'])['Total Amount of Assistance'].sum()
solar_inv = invest_df.loc[invest_df['Energy Source'] == 'Solar'].groupby(['State'])['Total Amount of Assistance'].sum()
geo_inv = invest_df.loc[invest_df['Energy Source'] == 'Geothermal'].groupby(['State'])['Total Amount of Assistance'].sum()
hydro_inv = invest_df.loc[invest_df['Energy Source'] == 'Hydroelectric'].groupby(['State'])['Total Amount of Assistance'].sum()


####################

#Turn the variables into a DataFrame of each Energy Source by State
energy_inv_total = pd.DataFrame({
                                "Wind": wind_inv,
                                "Biomass": bio_inv,
                                "Hydroelectric": hydro_inv,
                                "Solar": solar_inv,
                                "Geothermal": geo_inv})
                                
#Formatting columns for total amount of dollars invested
energy_inv_total['Wind'] = energy_inv_total['Wind']
energy_inv_total['Biomass'] = energy_inv_total['Biomass']
energy_inv_total['Hydroelectric'] = energy_inv_total['Hydroelectric']
energy_inv_total['Solar'] = energy_inv_total['Solar']
energy_inv_total['Geothermal'] = energy_inv_total['Geothermal']

energy_inv_total.reset_index()


###########################

#Replacing NaN's with 0 within DataFrame
energy_inv_total = energy_inv_total.fillna(0)
energy_inv_total.reset_index()


###############################


energy_inv_total['Wind'] = energy_inv_total['Wind'].astype(int)
energy_inv_total['Biomass'] = energy_inv_total['Biomass'].astype(int)
energy_inv_total['Hydroelectric'] = energy_inv_total['Hydroelectric'].astype(int)
energy_inv_total['Solar'] = energy_inv_total['Solar'].astype(int)
energy_inv_total['Geothermal'] = energy_inv_total['Geothermal'].astype(int)
energy_inv_total


##############################

#Create Hbar for each States total Investment for entire 2 decades per Energy Source

energy_inv_total.plot.bar(figsize =(25,15), title= 'Investment of Energy Source Per State',ylabel='Investment in Dollars (in Hundred Millions)')


##############################

bio_inv = energy_inv_total.sort_values(by=['Biomass'], ascending = False)
bio = list(bio_inv.iloc[0:5, 0].index)
geo_inv = energy_inv_total.sort_values(by=['Geothermal'], ascending = False)
geo = list(geo_inv.iloc[0:5, 1].index)
hyd_inv = energy_inv_total.sort_values(by=['Hydroelectric'], ascending = False)
hyd = list(hyd_inv.iloc[0:5, 2].index)
sol_inv = energy_inv_total.sort_values(by=['Solar'], ascending = False)
sol = list(sol_inv.iloc[0:5, 3].index)
wnd_inv = energy_inv_total.sort_values(by=['Wind'], ascending = False)
wnd = list(wnd_inv.iloc[0:5, 4].index)

# Create summary dataframe to show the top 5 states for each Energy Source
invest_summary = pd.DataFrame({'Rank': np.arange(1, 6, 1),
                        'Biomass ': bio,
                        'Geothermal ': geo,
                        'Hydroelectric ': hyd,
                        'Solar ': sol,
                        'Wind ': wnd})
invest_summary

################################

#Total amount Invested per year into each Energy Source
inv_df_bio = invest_df.loc[invest_df['Energy Source'] == source_list[0]].groupby(["Year"])['Total Amount of Assistance'].sum()
inv_df_geo = invest_df.loc[invest_df['Energy Source'] == source_list[1]].groupby(["Year"])['Total Amount of Assistance'].sum()
inv_df_hyd = invest_df.loc[invest_df['Energy Source'] == source_list[2]].groupby(["Year"])['Total Amount of Assistance'].sum()
inv_df_sol = invest_df.loc[invest_df['Energy Source'] == source_list[3]].groupby(["Year"])['Total Amount of Assistance'].sum()
inv_df_wnd = invest_df.loc[invest_df['Energy Source'] == source_list[4]].groupby(["Year"])['Total Amount of Assistance'].sum()

energy_total = pd.DataFrame({
                                "Wind": inv_df_bio,
                                "Biomass": inv_df_geo,
                                "Hydroelectric": inv_df_hyd,
                                "Solar": inv_df_sol,
                                "Geothermal": inv_df_wnd
                               
})

energy_total = energy_total.fillna(0)


#########################

#formatting
energy_total['Wind'] = energy_total['Wind'].astype(int)
energy_total['Biomass'] = energy_total['Biomass'].astype(int)
energy_total['Hydroelectric'] = energy_total['Hydroelectric'].astype(int)
energy_total['Solar'] = energy_total['Solar'].astype(int)
energy_total['Geothermal'] = energy_total['Geothermal'].astype(int)
energy_total.reset_index()

###############################

#Create Hbar graph showcase each years totals of Investment per Energy Source
energy_total.plot.bar(figsize =(20,10),title= 'Investment of Energy Source Per year',ylabel='Investment in Dollars (in Hundred Millions)')

###############################

# Group Production Data, add 'Energy Source' field and merge as one
prod_df_merge = prod_df.groupby(["Year", "State", "Energy Source"])['Generation (Megawatthours)'].sum()
prod_df_merge2 = prod_df.groupby(["Year", "State", "Energy Source"])['Energy Source'].first().reset_index(name ='Source')
prod_df_merge = pd.merge(prod_df_merge, prod_df_merge2, on = ["Year", "State", "Energy Source"])

# Group Investment Data
inv_df_merge = inv_df.groupby(["Year", "State", "Energy Source"])['Total Number of Investments', 'Total Amount of Assistance'].sum()

# Merge Production and Investment Data
temp = pd.merge(prod_df_merge, inv_df_merge, on = ["Year", "State", "Energy Source"])

# Merge Production, Investment, and Census Data
final = pd.merge(temp, cleaned_census_df, on = ["Year", "State"])

# Final data grouped by Year, State, Energy Source - reorder columns and add data fields
final = final[['Year', 'State', 'State Name', 'Population', 'Energy Source', 'Source', 'Generation (Megawatthours)', 'Total Number of Investments', 
               'Total Amount of Assistance']]
final['Generation per Investment'] = final['Generation (Megawatthours)'] / final['Total Amount of Assistance']
final['Generation per Capita'] = final['Generation (Megawatthours)'] / final['Population']
final['# Investments per Capita'] = final['Total Number of Investments'] / final['Population']
final['Investment per Capita'] = final['Total Amount of Assistance'] / final['Population']
final

###############################

# Population per State over all years
state_pop = cleaned_census_df.groupby('State')['Population'].sum()

# Production and Investment metrics per State over all years
final_state = final.groupby(['State'])[['Generation (Megawatthours)', 'Total Number of Investments', 'Total Amount of Assistance']].sum()

# Merge to get data grouped by state over all years and add variables
final_state = pd.merge(final_state, state_pop, on = "State")

final_state['Generation per Investment'] = final_state['Generation (Megawatthours)'] / final_state['Total Amount of Assistance']
final_state['Generation per Capita'] = final_state['Generation (Megawatthours)'] / final_state['Population']
final_state['# Investments per Capita'] = final_state['Total Number of Investments'] / final_state['Population']
final_state['Investment per Capita'] = final_state['Total Amount of Assistance'] / final_state['Population']
final_state.head()

###############################

# Generate a box plot of the Total Generation per Investment
y_axis1 = final_state['Generation per Investment']
y_axis2 = final_state['Generation per Capita']
y_axis3 = final_state['Investment per Capita']

fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
ax[0].set_ylabel('Generation per Investment')
ax[0].boxplot(y_axis1, flierprops = {'marker': 'o', 'markersize': 10, 'markerfacecolor': 'red'})
ax[0].set_xticklabels('')

ax[1].set_ylabel('Generation per Capita')
ax[1].boxplot(y_axis2, flierprops = {'marker': 'o', 'markersize': 10, 'markerfacecolor': 'red'})
ax[1].set_xticklabels('')

ax[2].set_ylabel('Investment per Capita')
ax[2].boxplot(y_axis3, flierprops = {'marker': 'o', 'markersize': 10, 'markerfacecolor': 'red'})
ax[2].set_xticklabels('')

plt.show()

###############################
