# Algebra-Solver.py
# Author : Aarthi Raghavendra
# Co-Worker : Jeremy Cox, Abhyudaya Upadhay
#
# Genetic algorithms for solving random algebraic equality relationships


from queue import PriorityQueue
import time
import copy
import sys
import math
import functools
import random


#check for powers and division by zero???

'''TREE FUNCTIONS'''

# depth 5 (bottom row) is always numbers
emptyTree = [' ' for x in range(31)]


# parent = int((node + 1)/2) -1 
def getParent ( index ):
    return int((index+1)/2)-1

def getLeft (index):
    if (index > 14):
        return -1
    else:
        return 2*index+1

def getRight (index):
    if (index > 14):
        return -1
    else:
        return 2*index+2


operators = [ '+', '-', '*', '/','**','S' ]
rowRange = [ (0,0), (0,0), (1,2), (3,6), (7,14), (15,30) ]

def sortMyList ( aList ):
    kevinbacon = PriorityQueue()
    result = []

    for k in range( len(aList) ):
        kevinbacon.put( copy.deepcopy(aList[k]) )

    for k in range( len(aList) ):
        result.append( copy.deepcopy(kevinbacon.get()) )

    return result


def randomTree2 ():
    return validateTree(  bigRandomTree( ), 0  )
    

def randomTree ( depth = 3 ):
    # depth from 1 thru 5
    # return a random tree

    result = emptyTree

    k=depth

    for j in range( rowRange[k][0], rowRange[k][1]+1 ):
        nug = int(1500 * random.random())
        nug -= 510
        if nug <= 510:
            result[j] = str(nug)
        else:
            result[j] = 'x'

        if result[j] == '0':
            result[j] = 'x'

    k -= 1
    
    while k > 0:

        for j in range( rowRange[k][0], rowRange[k][1]+1 ):
            result[j] = operators[ int(random.random() * len(operators) ) ]

            if result[j] == '**' and not isNum (result[getRight(j)]):
                result[j] = '+'
            if result[j] == '**' and isNum (result[getRight(j)]) and (not result[getRight(j)] == ' ')and int(result[getRight(j)]) > MAXEXPONENT:
            #elif result[index] == '**' and isNumOrX (result[getRight(index)]) and (not result[getRight(index)] == ' ') and int(result[getRight(index)]) > MAXEXPONENT:

                result[j] = '+'
            if result[j] == '/' and result[getRight(j)] == 'x':
                result[getRight(j)] = '3'
            if result[j] == 'S':
                if not result[getRight(j)] in operators:
                    result = writeToAllDaughterNodes ( result, 'x', getRight(j))
                #result[getLeft(j)]=' '
        
        k -= 1
    
    return result    


def bigRandomTree( ):
    # depth from 1 thru 5
    # return a random tree

    result = emptyTree

    #rowRange = [ (0,0), (0,0), (1,2), (3,6), (7,14), (15,30) ]

    for j in range( rowRange[5][0], rowRange[5][1]+1 ):
        nug = int(1500 * random.random())
        nug -= 510
        if nug <= 510:
            result[j] = str(nug)
        else:
            result[j] = 'x'

        if result[j] == '0':
            result[j] = 'x'

    result = randRecurse( result, 0 )

    return result

def randRecurse( result, index ):

    if( index < 15 ):
        if ( random.random() < RANDTREECHILDCHANCE ):
            result = randRecurse(result, getLeft(index) )
        else: 
            result = randRecurse(result, getRight(index) )     

    if( result[index] == ' ' ):
            result[index] = operators[ int(random.random() * len(operators) ) ]

            if result[index] == '**' and not isNum (result[getRight(index)]):
                result[index] = '+'
            elif result[index] == '**' and isNum (result[getRight(index)]) and (not result[getRight(index)] == ' ') and int(result[getRight(index)]) > MAXEXPONENT:
                result[index] = '+'
            #elif result[index] == '/' and result[getRight(index)] == 'x':
            #    result[getRight(index)] = '3'
            elif result[index] == 'S':
                if not result[getRight(index)] in operators:
                    result[getRight(index)]='x'

    return result    

#def 

def validateTree( aTree, index ):
    #fills in empty blanks with variables or numbers
    #checks that if depth is not 5, if bottom row numbers/variables has been changed, 
    
    if ( index < 15):
        aTree = validateTree( aTree, getRight(index) )
        aTree = validateTree( aTree, getLeft(index) )
        '''
        if (   aTree[index] in operators  and  isNum(aTree[getRight(index)])  and  isNum(aTree[getLeft(index)])  ):
            #evaluate and simplify expression; no more constants!!!
            print( "Simplifying tree: ", BtoEval(aTree), " AT index ", index, "--> operator ", aTree[index] )
            print( aTree)
            aTree = writeToAllDaughterNodes( copy.deepcopy(aTree), str(eval(BtoEval(aTree))), index )
            print( "Simplifyied tree: ", aTree)
        '''
    if (   ( aTree[index] == ' ' or isNumOrX(aTree[index]) )  and  index < 15   ):
        #numbers/variables above row 5 are forced to match row 5 upon validating
        #allows random generation to make sparse trees
        aTree[index] = copy.deepcopy( aTree[getLeft(index)] )
        
        '''
    if ( aTree[index] == ' ' ):
        aTree[index] = copy.deepcopy( aTree[getLeft(index)] )
        if ( random.random() < 0.50 ):
            aTree[index] = copy.deepcopy( aTree[getLeft(index)] )
        else:
            aTree[index] = copy.deepcopy( aTree[getRight(index)] )
        '''   

    return aTree


def BtoEval( bTree, x1 = -1000 ):
    #converts binary tree to algebra notation string
    #if x1 = -1000, returns expression
    #recursive (BAM!)
    x  = x1
    if (x1 == -1000):
        return recurse(bTree, 0, x1)
    else:
        newString = recurse(bTree, 0, x1)
        return eval( newString )



def BtoEvalDebug( bTree, x1 = -1000 ):
    #converts binary tree to algebra notation string
    #if x1 = -1000, returns expression
    #recursive (BAM!)
    print( "BtoEval stop: ", bTree)
    x  = x1
    if (x1 == -1000):
        return recurse(bTree, 0, x1)
    else:
        newString = recurse(bTree, 0, x1)
        return eval( newString )


def isNumOrX( aString, verbose = False ):

    if aString in operators or aString == ' ':
        if (verbose):
            print("isNumOrX checking string:>> ", aString, " :>> False")
        return False    
    else:
        if (verbose):
            print("isNumOrX checking string:>> ", aString, " :>> True")
        return True

def isNum( aString ):
    if aString in operators or aString == 'x' or aString == ' ':
        return False
    else:
        return True
   

def recurse (bTree, index, x1):
    if (bTree[index] == 'S'):
        return '(100*math.sin'+recurse(bTree, getRight(index), x1)+')'
    
    elif (bTree[index] == '/'):
        chump = recurse(bTree, getRight(index), x1)
        x = x1
        if ( not x1 == -1000 and eval(chump) == 0):
            return '('+recurse(bTree, getLeft(index), x1)+bTree[index]+'(1))'
        else:
            return '('+recurse(bTree, getLeft(index), x1)+bTree[index]+chump+')'

    elif (bTree[index] == '**'):
##catch error here
        chump = recurse(bTree, getRight(index), x1)
        x = x1
        if ( not x1 == -1000 and eval(chump) > 10):
            return '('+recurse(bTree, getLeft(index), x1)+bTree[index]+'(1))'
        else:
            return '('+recurse(bTree, getLeft(index), x1)+bTree[index]+chump+')'

    elif (bTree[index] in operators):
        return '('+recurse(bTree, getLeft(index), x1)+bTree[index]+recurse(bTree, getRight(index),x1)+')'
    else:
        return '('+bTree[index]+')'

'''END TREE FUNCTIONS'''



def tuplize( aString ):
    temp = aString.lstrip( " (" )
    temp = temp.rstrip( " )" )
    temp = temp.split( "," )
    ret = ( int(temp[0]), int(temp[1]) )
    return ret


def checkFit( equationTree, Xarray, Yarray ):

    Zarray = []
    for k in range(1000):
        x = Xarray[k]
        Zarray.append( BtoEval(equationTree, k) - Yarray[k] )

    mean = sum(Zarray)/len(Zarray)
    variance = 0
    for k in range(len(Zarray)):
        variance += (Zarray[k]-mean)**2 / (VARIANCECUTTER**2 * len(Zarray))
    rms = math.sqrt(variance) *  MULTIPLYRATIO
    return (rms, mean)


def scalarChild( bTree, Xarray, Yarray ):
    #tree must be depth 4 or less
    #creates a child with perfect scalar correction

    result = copy.deepcopy(emptyTree)
    k = 0
    roots = 0
    Zarray = []

    for k in range(1,1000):
        x = Xarray[k]
        check = BtoEval(bTree, k)
        if ( not check == 0 ):
            Zarray.append( Yarray[k] / check )

    #print( "basic tree: ", bTree)
    #print( "basic tree: ", BtoEval(bTree) )

    mean = sum(Zarray) / len(Zarray)
    #print( "m: ", mean, " s: ", sum(Zarray), " l: ", len(Zarray) )

    posMean = mean
    if posMean < 0:
        posMean = -1 * posMean

    index = math.log(posMean)
   
    if (index < 0 and roots < 3):
        #if a divison scalar
        k = 1 / posMean
        while k > 510:
            k = int(math.sqrt(k))
            roots += 1
        k = round(k)
        result[0] = '/'
        result = copy.deepcopy( subTreeCopy(  result, copy.deepcopy(result), bTree, 1, 0 ) )  #copy tree to left half of new tree
        result = writeToAllDaughterNodes( result, str(k), 2 )
        if ( mean < 0 ):
            result[2] = str(-1*k)
            result[5] = str(-1*k)
            result[11] = str(-1*k)
            result[23] = str(-1*k)
        if ( roots > 0 ):
            result[2] = '*'
        if ( roots > 1 ):
            result[5] = '*'
            result[6] = '*'
        if ( roots > 2 ):
            result[11] = '*'
            result[12] = '*'
            result[13] = '*'
            result[14] = '*'
    else:
        k = int(100*mean)
        result[0] = '*'
        result = copy.deepcopy( subTreeCopy( copy.deepcopy(result), result, bTree, 2, 0 ) )   #copy tree to right half of
        result[1] = '/'
        result = writeToAllDaughterNodes( result, str(k), 3 )
        result = writeToAllDaughterNodes( result, '100', 4 )
        
    result = validateTree( result, 0 )
    #print( "created scalar tree: ", result)
    print( "created scalar tree: ", BtoEval(result) )
    #print( "******")
    #print( " ")

    return result
    #return validateTree( result, 0 )
    
def treeDepth( bTree ):
    #if operators are in row 4, has depth 5, etc
    #if blank, returns -1

    result = -1   #sentinel

    if isNumOrX( bTree[0] ):
        print( "depth 1 found" )
        result = 1
    else:
        row = [[] for k in range(6)]
        row[1] = bTree[0:1]
        row[2] = bTree[1:3]
        row[3] = bTree[3:7]
        row[4] = bTree[7:15]
        row[5] = bTree[15:31]

        print(" ")
        for k in range(1, 5+1):
            z = 6 - k
            print("looking in row", z, " for operators: ", row[z])
            for j in range( len(row[z]) ):
                for m in range( len(operators) ):
                    if ( z+1 > result and operators[m] in row[z] ):
                        result = z+1

    return result
    
def newChild( evalChromos, numBest ):

        if ( len (evalChromos) < numBest ):
            numBest = evalChromos
    
        k = int(random.random() * numBest)
        j = int(random.random() * numBest)
        while j == k:
            j = int(random.random() * numBest)
        #result = copy.deepcopy(evalChromos[k][2])

        subTreeTarget = int( 29 * random.random() ) + 1
        subTreeSource = int( 29 * random.random() ) + 1
        while subTreeTarget > subTreeSource :
                subTreeSource = int( 29 * random.random() ) + 1
        #print("k: ",evalChromos[k][2],"  j: ",evalChromos[k][2])
        #print("crossing ",BtoEval(evalChromos[k][2])," with ", BtoEval(evalChromos[j][2]), " at ", subTreeTarget, " <-- ", subTreeSource)

        result = copy.deepcopy(  subTreeCopy( copy.deepcopy(evalChromos[k][2]),evalChromos[k][2], evalChromos[j][2], subTreeTarget, subTreeSource )  )
        
        print( " >>>> ", BtoEval(result) )

        return result

def writeToAllDaughterNodes ( bTree, newString, index ):
    result = copy.deepcopy( bTree )
    #print( "writing to node ", index )
    #print( 'bTree[index] is ', bTree[index])
    result[index] = copy.deepcopy( newString )
    if (index < 15):
        result = writeToAllDaughterNodes( result, newString, getRight(index) )
        result = writeToAllDaughterNodes( result, newString, getLeft(index) )

    return result   

def subTreeCopy( result, treeA, treeB, tarA, tarB ):
        #intial call to subTreeCopy passes copy of treeA as result
        
        #if (   int(math.log( tarA+1, 2))   >   int(math.log( tarB+1, 2))   ):
        #    print("WARNING subTreeCopy called to target tree at level ", int(math.log( tarA+1, 2)) +1, "and source at level ", int(math.log( tarB+1, 2))+1)
        #    print("WARNING the copied tree will be abridged")

        #print( "tarA:  ", tarA, "  tarB:  ", tarB )
        #result = copy.deepcopy( treeA )

        if ( isNumOrX(treeB[tarB]) ):
            #cannot get here with invalid tarA value
            #print( "copying from treeB at ", tarB, " : string >", treeB[tarB], "< as a leaf to treeA at ", tarA)
            result = writeToAllDaughterNodes( result, copy.deepcopy( treeB[tarB]), tarA )
            #print( "result: ", result )
            #print( "treeB:  ", treeB )
        elif ( treeB[tarB] in operators and tarA < 15):
            result[tarA] = copy.deepcopy( treeB[tarB] )
            #print( "copying subtree with children.... at ", tarB, " : string >", treeB[tarB], "< as a non-leaf to treeA at ", tarA)
            #print( "result: ", result )
            #print( "treeB:  ", treeB )
            #print( "       >>>>>>>>>>>recurse until leaf nodes found....")
            result = subTreeCopy ( copy.deepcopy(result), treeA, treeB, getLeft(tarA), getLeft(tarB) )
            result = subTreeCopy ( copy.deepcopy(result), treeA, treeB, getRight(tarA), getRight(tarB) )

        elif ( treeB[tarB] in operators and tarA > 14 and tarB < 15):
            #abridge result; parse down until you find right leaf in treeB
            result = subTreeCopy ( result, treeA, treeB, tarA, getRight(tarB) )
        else:
            treeA[tarA] = '1'

        return result

def ComboScalar( f1, f2, operator, Xarray, Yarray ):
    #creates a scalar fit   m*f1 (operator) m*f2
    #f1, f2, and result are trees
    # as x increase, d/dx f1 >>  d/dx f2

    result = copy.deepcopy( emptyTree )
    result[0] = copy.deepcopy(operator)
    
    newf1 = scalarChild( f1, Xarray, Yarray )
    newf1 = validateTree(newf1, 0)
    result = subTreeCopy( result, result, newf1, 1, 0 )
    
    Zarray = []  #we are skipping x=0
    for k in range(0,1000):
        x = Xarray[k]
        Zarray.append( BtoEval(newf1, k) - Yarray[k] )

    newf2 = scalarChild( f2, Xarray, Zarray )
    newf2 = validateTree(newf2, 0)
    result = subTreeCopy( result, result, newf2, 2, 0 )

    result = validateTree( result, 0 )
    #print(  "functions: ", BtoEval(newf1), "  ", operator, "  ", BtoEval(newf2)  )
    print( "created combo tree: ", BtoEval(result) )

    return result



    
def printResult( aTuple ):
    result = False
    print( "RMS=","{0:.2f}".format(aTuple[0]*RMSCUTOFF),", for ",BtoEval(aTuple[2])," + ", "{0:.0f}".format(aTuple[1]) )
    if ( aTuple[0] < 1 ):
        print ( "***** This meets or exceeds RMS maximum of ", RMSCUTOFF, " *******" )
        result = True

    return result


#GLOBAL CONSTANTS
PRIMITIVES = []
PRIMITIVES.append (['/', 'x', '503'] + ['1' for k in range(28)] )
PRIMITIVES.append (['/', 'x', '101'] + ['1' for k in range(28)] )
PRIMITIVES.append (['/', 'x', '17'] + ['1' for k in range(28)] )
PRIMITIVES.append (['S', 'x', 'x'] + ['1' for k in range(28)] )
PRIMITIVES.append (['x'] + ['1' for k in range(30)] )
PRIMITIVES.append (['**', 'x', '2'] + ['1' for k in range(28)] )
PRIMITIVES.append (['**', 'x', '3'] + ['1' for k in range(28)] )
PRIMITIVES.append (['**', 'x', '4'] + ['1' for k in range(28)] )




#GLOBAL CONSTANTS
# >>> RUN TIME BEHAVIOR CONTROL <<<
SIZESPACE = 40
MAXGENERATIONS = 20
   
WINNERS = int(SIZESPACE * 0.10)
DELETES = 2 * WINNERS

MAXEXPONENT = 10

RANDTREECHILDCHANCE = 0.70

RMSCUTOFF = 5000
VARIANCECUTTER = 1000000
MULTIPLYRATIO = VARIANCECUTTER / RMSCUTOFF


if __name__ == "__main__":

    '''
    ### TEST CODE FOR BASIC FUNCTIONS ###
    bacon = bigRandomTree( )
    print ("RandTree: ", bacon)
    print ("RandTree: ", BtoEval(bacon) )
    bacon = validateTree( bacon, 0 )
    print ("validated: ", bacon )
    print ("validated: ", BtoEval(bacon) )
    print ("depth: ", treeDepth(bacon) )

    kevin = copy.deepcopy( emptyTree )
    kevin[0] = 'x'
    print ("making depth 1:  ", kevin)
    kevin = validateTree( kevin, 0 )  
    print ("tree :", BtoEval(kevin), "  depth: ", treeDepth(kevin) )
    
    for k in range(1,6):
        print ( "**********************************" )
        kevin = copy.deepcopy( randomTree(k) )
        print ("making depth ", k,":  ", kevin)
        kevin = validateTree( kevin, 0 )  
        print ("tree :", BtoEval(kevin), "  depth: ", treeDepth(kevin) )
    
    print("This good for nothing   subTreeCopy()  ain't working")

    temp = copy.deepcopy( emptyTree )   
    temp = copy.deepcopy( subTreeCopy( copy.deepcopy(temp), temp, PRIMITIVES[1], 1, 1) )
    print("subTree copy into empty(left): ", temp )
    '''


   ##################################
   #GLOBAL RUNTIME CONSTANTS        #
   #                                #
    SIZESPACE = 40
    MAXGENERATIONS = 20
    WINNERS = 15
    DELETES = 15
   #                                #
   ##################################

    time.clock()
    
    temp = copy.deepcopy( emptyTree )

    Xarray = []
    Yarray = []

    #file = open(sys.argv[1])
    file = open("fn3.csv")
    for i in range(0,1000):
        (xi, yi) = tuplize( file.readline() )
        Xarray.append( xi )
        Yarray.append( yi )

    evalChromos = []

    searchON = True  #bool for solution not found

    print( "Generating scalar primitive good fits.... ", "{0:.2f}".format(time.clock()), " s elapsed. " )    
    for k in range( 0, len(PRIMITIVES) ):
        temp = scalarChild( PRIMITIVES[k], Xarray, Yarray )
        result = checkFit( temp, Xarray, Yarray )
        evalChromos.append( (result[0],result[1], copy.deepcopy(temp)) )
    

    print( "Generating scalar primitive COMBO good fits.... ", "{0:.2f}".format(time.clock()), " s elapsed. " )   
    for k in range( 4, len(PRIMITIVES) ):
        temp = ComboScalar( PRIMITIVES[k],PRIMITIVES[3], '+', Xarray, Yarray )
        #print( "KKK_COMBO is ", temp )
        result = checkFit( temp, Xarray, Yarray )
        evalChromos.append( (result[0],result[1], copy.deepcopy(temp)) )
        for j in range(4, k):
            temp = ComboScalar( PRIMITIVES[k],PRIMITIVES[j], '+', Xarray, Yarray )
            result = checkFit( temp, Xarray, Yarray )
            evalChromos.append( (result[0],result[1], copy.deepcopy(temp)) )            

#def ComboScalar( f1, f2, operator, Xarray, Yarray ):
        
    '''
    print( "Generating generation ", z, " random functions.... ", "{0:.2f}".format(time.clock()), " s elapsed. " )
    print( "        Processing, please wait ---->" )
    for k in range(len(PRIMITIVES)+1, SIZESPACE ):
        temp = randomTree2()
        #print( temp, ' ><>< ', BtoEval(temp) )
        result = checkFit( temp, Xarray, Yarray )
        evalChromos.append( (result[0],result[1], copy.deepcopy(temp)) )
    '''

    #delete losers
    evalChromos = sortMyList(evalChromos)[:(SIZESPACE-DELETES)]


    searchOn = False

            
    z = 0
    while( searchON and z < MAXGENERATIONS ):   


        #OUTPUT generation leaders
        print ('XXXXX')
        print ( "time:       ", "{0:.2f}".format(time.clock()), " s elapsed. ")
        print ( "generation: ", z, "   leaderboard::(the top ", WINNERS, ") " )
        print ('XXXXX')     

        for k in range( min(WINNERS, len(evalChromos)) ):
            if (  printResult( evalChromos[k] )  ):
                searchON = False

        #create and evaluate new children
        if ( searchON ):
            print( "Generating generation ", z, " random functions.... ", "{0:.2f}".format(time.clock()), " s elapsed. " )
            for k in range(DELETES):
                temp = copy.deepcopy( newChild( evalChromos, WINNERS ) )
                result = checkFit( temp, Xarray, Yarray)
                evalChromos.append( (result[0], result[1], copy.deepcopy(temp) ) )

        #delete losers
        if ( searchON ):
            evalChromos = sortMyList(evalChromos)[:(SIZESPACE-DELETES)]

        z += 1
        #end while searchON

