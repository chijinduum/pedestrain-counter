# https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/

from json_minify import json_minify
import json

class TrackableObject:    
    def __init__(self, objectID, centroid):
        """
        store the objectID, then initialize a list of centroids
        using the current centroid
        """
        self.objectID = objectID
        self.centroids = [centroid]
        

        self.counted = False
        
        # initialize the dictionaries to store the timestamp and 
        # position of the object at various point
        
        self.timestamp = {"A": 0, "B": 0, "C": 0, "D": 0}            
        self.position = {"A": None, "B": None, "C": None, "D": None}
        self.lastPoint = False
        
        # initialise the object speeds in MPH and KMPH
        self.speedMPS = None
        self.speedKMPH = None
        
        
        # initialize two boolean, (1) used to indicate if the object's speed
        # has already been estimated or not, and (2) used to indivate 
        # if the object's speed has been logged or not
        
        self.estimated = False
        self.logged = False
        
        # initialize the direction of the object
        self.direction = None

    def calculate_speed(self, estimatedSpeeds):
        # calculate the speed in KMPH and MPH
        self.speedKMPH = np.average(estimatedSpeeds)
        METERS_PER_SECOND = 0.277778
        self.speedMPS = self.speedKMPH * METERS_PER_SECOND

class Conf:
    def __init__(self, confPath):
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)
    def __getitem__(self,k):
        # return the value associate with the supplied key
            return self.__dict__.get(k, None)