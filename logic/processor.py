import requests
import geocoder
from geopy.geocoders import Nominatim

class Response:
    def __init__(self, text=None, payload=None, status='ok'):
        self.text = text
        self.payload = payload
        self.status = status

    def __repr__(self):
        return f"[{self.status}] {self.text}"

    def __str__(self):
        return self.__repr__()


def _get_intent(text):
    """
    This function should compute the intent of the request, and then in will be passed to the corresponding service
    """
    if 'coffee' in text:
        return 'coffee'
    if 'translate' in text:
        return 'translate'
    if 'weather' in text:
        return 'weather'

    #TODO: implement

    return None



def _coffee(text):
    return Response(text=f"Cannot perform `{text}` - I'm a teapot")


def _translate(text):
    return Response(text="")

    
def temperature_conversion(temp, logging):
    '''
    #Author: Nils Becker
    Converts a given temperature in °K into a °C String representation
    :param temp: Float or int Value of temperature in °K
    :param logging: Currently used instance of logging module
    :returns: The temperature converted to °C as a String
    '''
    try:
        return f"{int(temp)-273}°C"
    except ValueError:
        logging.error(f"{temp} must be an int or float")
        pass

def _weather(cfg, logging):
    '''
    #Author: Nils Becker
    Gives a weather forecast for a specified city
    :param cfg: Currently used configuration
    :param logging: Currently used instance of logging module
    :returns: weather forecast for now/ next 12h/ next 7 days in a List
    '''
    ip = geocoder.ip("me")
    location = ip.latlng
    locator = Nominatim(user_agent="Larissa")
    try:
        #get city and country from ip adress
        loc = locator.reverse(f'{location[0]},{location[1]}').raw['address']
        city = loc.get('city', '')
        country = loc.get('country', '')
        if city == "":
            city = loc.get('town')
            if city == "":
                city = location
    except AttributeError:
        logging.error(f'No data for coordinates {location} found!')
        return Response(f'No data for coordinates {location} found!')
    
    #get key from https://openweathermap.org/
    api_key = cfg['weather']['apikey']
    try:
        #create API request for given city
        url = 'https://api.openweathermap.org/data/2.5/onecall?lat={}&lon={}&exclude={}&appid={}'.format(location[0], location[1], "minutely,alerts", api_key)
        response = requests.get(url).json()
    except AttributeError:
        logging.error(f"City {city} not found!")
        return Response(text = f"City {city} not found!")
    result = []
    try:
        #get data
        current = response['current']
        currentList = []
        temperature = temperature_conversion(current['temp'], logging)
        weather = current['weather']
        description = weather[0]['description']
        clouds = f"{current['clouds']}% clouds"
        currentList.append(temperature)
        currentList.append(description)
        currentList.append(clouds)
        result.append([currentList])
        for option in ["hourly", "daily"]:
            #get hourly, then daily data

            currentList = []
            current = response[option]
            #if option == hourly, the next 12 hours are relevant -> length = 13
            #if option == daily, the next 7 days are relevant -> length 8
            length = 13 if option == "hourly" else 8 
            for i in range (1, length):
                #loop through the next 12 hours or next 7 days

                temporary = current[i]
                loopList = [i]
                if option == "daily":
                    temp = temporary['temp']
                    average = f"{temperature_conversion(temp['day'], logging)} average"
                    min = f"{temperature_conversion(temp['min'], logging)} minimum"
                    max = f"{temperature_conversion(temp['max'], logging)} maximum"
                    loopList.append(average)
                    loopList.append(min)
                    loopList.append(max)
                else:
                    temperature = temperature_conversion(temporary['temp'], logging)
                    loopList.append(temperature)
                weather = temporary['weather']
                description = weather[0]['description']
                clouds = f"{temporary['clouds']}% clouds"
                pop = f"{temporary['pop']*100}% chance of precipation"
                loopList.append(description)
                loopList.append(clouds)
                loopList.append(pop)
                currentList.append(loopList)
            result.append(currentList)
    except KeyError as e:
        logging.error(f"{e} Weather API Error: Field {e} not available")
        return Response(f"{e} Weather API Error: Field {e} not available")
    except TypeError as e:
        logging.error(f"Weather API Error: Field {e} returned None")
        return Response(text = "Weather API Error: Field {e} returned None") 
       
    text = f" \nWeather forecast for {city}, {country}:\n\nNow:\n"
    for content1 in result[0]:
        for content2 in content1:
            text += f"{content2}, "  
    text = f"{text}\n\nNext 12h:\n"
    for content1 in result[1]:
        for content2 in content1:
            text = f"{text} {content2}, "
        text += "\n"
    text = f"{text}\nNext 7 days:\n"
    for content1 in result[2]:
        for content2 in content1:
            text = f"{text} {content2}, "
        text += "\n"
    return Response(text = text)

def process(command, cfg, logging):
    logging.info(f"Processing command `{command}`")
    intent = _get_intent(command)
    if 'coffee' == intent:
        return _coffee(command)
    elif 'translate' == intent:
        return _translate(text)
    elif 'weather' == intent:
        return _weather(cfg, logging)

    return None
