from functools import cache
from frozendict import frozendict

from li_basic_utilities import StaticClass


class CocoExtensionForChair(StaticClass):
    # https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py
    @cache
    @staticmethod
    def synonyms() -> frozendict[str, str]:
        # https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
        raw: str = """person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush"""

        synonyms_map: dict[str, str] = {}
        line: str
        for line in raw.splitlines():
            synonyms: list[str] = [s.strip() for s in line.split(",")]
            synonym: str
            for synonym in synonyms:
                synonyms_map[synonym] = synonyms[0]
        return frozendict(synonyms_map)
    
    @staticmethod
    def __double_words():
        return ("motor bike", "motor cycle", "air plane", "traffic light", 
                "street light", "traffic signal", "stop light", "fire hydrant", 
                "stop sign", "parking meter", "suit case", "sports ball", 
                "baseball bat", "baseball glove", "tennis racket", "wine glass", 
                "hot dog", "cell phone", "mobile phone", "teddy bear", "hair drier", 
                "potted plant", "bow tie", "laptop computer", "stove top oven", 
                "hot dog", "teddy bear", "home plate", "train track")
    
    @staticmethod
    def __animal_words():
        return ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", 
                "bear", "zebra", "giraffe", "animal", "cub")
    
    @staticmethod
    def __vehicle_words():
        return ("jet", "train")
    
    @cache
    @staticmethod
    def double_words() -> frozendict[str, str]:
        result: dict[str, str] = {}

        double_word: str
        for double_word in CocoExtensionForChair.__double_words():
            result[double_word] = double_word

        animal_word: str
        for animal_word in CocoExtensionForChair.__animal_words():
            result[f"baby {animal_word}"] = animal_word
            result[f"adult {animal_word}"] = animal_word

        vehicle_word: str
        for vehicle_word in CocoExtensionForChair.__vehicle_words():
            result[f"passenger {vehicle_word}"] = vehicle_word
            
        result["bow tie"] = "tie"
        result["toilet seat"] = "toilet"
        result["wine glas"] = "wine glass"
        return frozendict(result)
