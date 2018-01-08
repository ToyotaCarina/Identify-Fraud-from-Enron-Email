import matplotlib.pyplot as plt
from beautifultable import BeautifulTable
from feature_format import featureFormat

def explore_dataset(data_dict):
    print("Number of rows: " + str(len(data_dict.keys())))
    print("Number of features: " + str(len(data_dict[data_dict.keys()[0]])))
    print("Number of POIs: " + str(sum(data_dict[person_name]['poi'] for person_name in data_dict.keys())))

def counting_NaNs(data_dict, features_list):
    print("\nMissing values by feature:")
    table = BeautifulTable()
    table.column_headers = ["feature", "count", "POI", "Non-POI"]
    for feature in features_list:
        if feature <> 'poi':
            counter_POI = sum(data_dict[each][feature] == 'NaN' and data_dict[each]['poi'] == 1 for each in data_dict)
            counter_nonPOI = sum(data_dict[each][feature] == 'NaN' and data_dict[each]['poi'] == 0 for each in data_dict)
            total_NaN = counter_POI + counter_nonPOI
            counter_POI_pcent = round(counter_POI / float(total_NaN) * 100,2)
            counter_nonPOI_pcent = 100 - counter_POI_pcent
            table.append_row([feature, total_NaN , str(counter_POI_pcent) + "%", str(counter_nonPOI_pcent) + "%"])
    print(table)

def draw_scatter_plot(data_dict, feature_1, feature_2):
    data = featureFormat(data_dict, [feature_1, feature_2])
    for p in data:
        x = p[0]
        y = p[1]
        plt.scatter(x, y)
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.show()

def find_empty_rows(data_dict):
    for key in data_dict.keys():
        all_rows_NaN = True
        for feature in data_dict[key] :
            if feature <> 'poi':
                all_rows_NaN = (all_rows_NaN) and (data_dict[key][feature] == 'NaN')         
        if all_rows_NaN :
            print 'Empty data for person:' + key    
       
        
        
