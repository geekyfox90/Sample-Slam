cmake_minimum_required(VERSION 3.7.2)

##################################################
set (VERSION_NUMBER "0.6.0")
project("PipelineSlam")
set (SOURCES 
    src/PipeLineSlam.cpp
	src/component.cpp
)

set (HEADERS
    interfaces/PipeLineSlam.h
)

##################################################

# various macros
include("$ENV{BCOMDEVROOT}/bcomBuild/SolARFramework/solarmacros.cmake")
# config setup
setup()
# process packagedependencies.txt
processPackagedependencies()
# define targets (type library)
defineTargets("library" "")



