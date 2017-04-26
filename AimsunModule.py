# -*- coding: utf-8 -*-
"""
Created on Sun Dec 04 14:58:57 2016

@author: Benjamin
"""

import sys, socket, json
from threading import Thread
from PyANGBasic import *
from PyANGKernel import *
from PyANGConsole import *
from PyMesoPlugin import *

class ListeningThread(Thread):
    socket = None
    aimsunInterface = None

    def __init__(self, aimsunInterface):
        Thread.__init__(self)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect(('localhost', 2199))
        self.aimsunInterface = aimsunInterface
        return

    def run(self):
        while True:
            buff = self.socket.recv(1024)
            rawData = json.loads(buff)
            requestType = rawData[0]
            if requestType == "SIMULATE":
                self.aimsunInterface.runSimulation()
            if requestType == "SET":
                params = rawData[1:]
                self.aimsunInterface.setParams(params)
            if requestType == "INTERRUPT":
                break
            self.socket.send("DONE")
            print('Request executed.')
        return


""" 
This class implements the interface between Python and Aimsun. It allows us to directly call two important
functions : 
    -runSimulation, which calls for a simulation to be run.
    -setParams, which modifies the parameters inside the simulation model.
"""

class AimsunInterface:
    console = None
    model = None
    replication = None
    simulator = None
    plugin = None

    def __init__(self, path, replication):
        self.console = ANGConsole()
        # Load a network
        if not self.console.open(path):
            print("Error loading file.")
            return
        print("File opened.")
        self.model = self.console.getModel()
        self.replication = self.model.getCatalog().find(replication)
        self.plugin = GKSystem.getSystem().getPlugin( "AMesoPlugin" )
        self.simulator = self.plugin.getCreateSimulator( self.model )
        return

    def runSimulation(self):
        """ This function runs the simulation in Aimsun.
        """
        self.simulator.addSimulationTask( GKSimulationTask(self.replication,GKReplication.eBatch) )
        self.simulator.simulate()

        return

    def setParams(self, params):
        """ This function modifies the parameters inside the Aimsun model. The four parameters we consider here
            are in order: 
                -reactionTimeAtt
                -reactionTimeAtTrafficLightAtt 
                -lookaheakDistance
                -reactionTimeVariationAtt (adjusted for each section)
        """
        print("Setting params...")
        reactionTime = params[0]
        reactionTimeAtTrafficLight = params[1]
        lookAheadDistance = params[2]
        reactionTimeVariationFactor = params[3:]

        if self.replication.getExperiment().getSimulatorEngine() == GKExperiment.eMeso:
            
            experimentType = self.model.getType( "GKExperiment" )
            for experiment in self.model.getCatalog().getObjectsByType(experimentType).itervalues():
                """ Modification of reactionTimeAtt. """
                experiment.setDataValueByID(GKExperiment.reactionTimeAtt, QVariant(reactionTime))
                """ Modification of reactionTimeAtTrafficLight. """
                experiment.setDataValueByID(GKExperiment.reactionAtTrafficLightMesoAtt, QVariant(reactionTimeAtTrafficLight))

            """ Modification of LookaheadDistance. """
            sectionType = self.model.getType( "GKSection" )
            lookaheadColumn = self.model.getColumn("GKTurning::lookaheadDistanceAtt");
            for section in self.model.getCatalog().getObjectsByType(sectionType).itervalues():
                for turn in section.getDestTurnings():
                    turn.setDataValueDouble(lookaheadColumn,lookAheadDistance)
            
            """ Modification of reactionTimeFactorAtt by section."""
            sections = list(self.model.getCatalog().getObjectsByType(sectionType).itervalues())
            print("Sections:", len(sections))
            for i in range(len(sections)):
                sections[i].setDataValueByID(GKSection.reactionTimeFactorAtt, QVariant(reactionTimeVariationFactor[i]))

        return

def main(argv):
    path = argv[1]
    replication = int(argv[2])
    aimInter = AimsunInterface(path, replication)
    listeningThread = ListeningThread(aimInter)
    listeningThread.start()

if __name__ == "__main__":
     sys.exit(main(sys.argv)) 