# coding=utf-8
"""Tarea 3"""

import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from operator import add

__author__ = "Ivan Sipiran"
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.viewPos = np.array([2.0,1.5,8.5]) # posicion inicial de la camara
        self.at = np.array([2.0,1.0,5.0]) # posicion inicial hacia donde mira
        self.camUp = np.array([0, 1, 0]) # el "arriba" de la camara
        self.car = np.array([2.0,0.0,5.0]) # posicion del auto
        self.dir = np.array([0.0,0.0,-1.0]) # direccion en la que apunta la punta del auto
        self.speed = 0.0075 # velocidad con la que se va a mover el auto
        self.rotSpeed = np.pi/900 # velocidad con la que se gira el auto


controller = Controller()

def setPlot(texPipeline, axisPipeline, lightPipeline):
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)
    
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "lightPosition"), 5, 5, 5)
    
    glUniform1ui(glGetUniformLocation(lightPipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "constantAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(lightPipeline.shaderProgram, "quadraticAttenuation"), 0.01)

def setView(texPipeline, axisPipeline, lightPipeline):
    view = tr.lookAt(
            controller.car - controller.dir* 3.5 + np.array([0.0,1.5,0.0]), # la camara siempre mirara desde detras y arriba de la posicion del auto
            controller.car + np.array([0.0,1.0,0.0]), # mira hacia un poco mas arriba del auto
            controller.camUp 
            
        )

    glUseProgram(axisPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(texPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(texPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(lightPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(lightPipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(lightPipeline.shaderProgram, "viewPosition"), controller.viewPos[0], controller.viewPos[1], controller.viewPos[2])
    

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return
    
    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_LEFT_CONTROL:
        controller.showAxis = not controller.showAxis

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
    
        
    elif key != glfw.KEY_W and key != glfw.KEY_A and key != glfw.KEY_S and key != glfw.KEY_D:
        print('Unknown key')

def createOFFShape(pipeline, filename, r,g, b):
    shape = readOFF(getAssetPath(filename), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()
            
            indices += [index, index + 1, index + 2]
            index += 3        



        return bs.Shape(vertexDataF, indices)

def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

def createTexturedArc(d):
    vertices = [d, 0.0, 0.0, 0.0, 0.0,
                d+1.0, 0.0, 0.0, 1.0, 0.0]
    
    currentIndex1 = 0
    currentIndex2 = 1

    indices = []

    cont = 1
    cont2 = 1

    for angle in range(4, 185, 5):
        angle = np.radians(angle)
        rot = tr.rotationY(angle)
        p1 = rot.dot(np.array([[d],[0],[0],[1]]))
        p2 = rot.dot(np.array([[d+1],[0],[0],[1]]))

        p1 = np.squeeze(p1)
        p2 = np.squeeze(p2)
        
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])
        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        
        indices.extend([currentIndex1, currentIndex2, currentIndex2+1])
        indices.extend([currentIndex2+1, currentIndex2+2, currentIndex1])

        if cont > 4:
            cont = 0


        vertices.extend([p1[0], p1[1], p1[2], 0.0, cont/4])
        vertices.extend([p2[0], p2[1], p2[2], 1.0, cont/4])

        currentIndex1 = currentIndex1 + 4
        currentIndex2 = currentIndex2 + 4
        cont2 = cont2 + 1
        cont = cont + 1

    return bs.Shape(vertices, indices)

def createTiledFloor(dim):
    vert = np.array([[-0.5,0.5,0.5,-0.5],[-0.5,-0.5,0.5,0.5],[0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0]], np.float32)
    rot = tr.rotationX(-np.pi/2)
    vert = rot.dot(vert)

    indices = [
         0, 1, 2,
         2, 3, 0]

    vertFinal = []
    indexFinal = []
    cont = 0

    for i in range(-dim,dim,1):
        for j in range(-dim,dim,1):
            tra = tr.translate(i,0.0,j)
            newVert = tra.dot(vert)

            v = newVert[:,0][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 1])
            v = newVert[:,1][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 1])
            v = newVert[:,2][:-1]
            vertFinal.extend([v[0], v[1], v[2], 1, 0])
            v = newVert[:,3][:-1]
            vertFinal.extend([v[0], v[1], v[2], 0, 0])
            
            ind = [elem + cont for elem in indices]
            indexFinal.extend(ind)
            cont = cont + 4

    return bs.Shape(vertFinal, indexFinal)

# TAREA3: Implementa la función "createHouse" que crea un objeto que representa una casa
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createHouse(pipeline):
    wall = createGPUShape(pipeline, bs.createTextureQuad(2.0,2.0))
    wall.texture = es.textureSimpleSetup( # aca se implementan las texturas de las paredes de la casa
        getAssetPath("wall4.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)


    roof = createGPUShape(pipeline, bs.createTextureQuad(2.0,2.0))
    roof.texture = es.textureSimpleSetup( # y aca las del techo
        getAssetPath("roof5.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)
    

    # y se colocan para formar la casa

    wall1Node = sg.SceneGraphNode('wall1')
    wall1Node.transform = tr.translate(0,0,0.5)
    wall1Node.childs += [wall]


    wall2Node = sg.SceneGraphNode('wall2')
    wall2Node.transform = tr.translate(0,0,-0.5)
    wall2Node.childs += [wall]


    wall3Node = sg.SceneGraphNode('wall3')
    wall3Node.transform = tr.matmul([tr.translate(0.5,0,0), tr.rotationY(np.pi/2)])
    wall3Node.childs += [wall]


    wall4Node = sg.SceneGraphNode('wall4')
    wall4Node.transform = tr.matmul([tr.translate(-0.5,0,0), tr.rotationY(np.pi/2)])
    wall4Node.childs += [wall]


    wall5Node = sg.SceneGraphNode('wall5')
    wall5Node.transform = tr.matmul([tr.translate(0,0.5,0.49), tr.rotationZ(np.pi/4), tr.scale(1/np.sqrt(2), 1/np.sqrt(2),1/np.sqrt(2))])
    wall5Node.childs += [wall]


    wall6Node = sg.SceneGraphNode('wall6')
    wall6Node.transform = tr.matmul([tr.translate(0,0.5,-0.49), tr.rotationZ(np.pi/4), tr.scale(1/np.sqrt(2), 1/np.sqrt(2),1/np.sqrt(2))])
    wall6Node.childs += [wall]


    roof1Node = sg.SceneGraphNode('roof1')
    roof1Node.transform = tr.matmul([tr.scale(1,1,1.15),tr.translate(1/3,2/3,0), tr.rotationZ(np.pi/4) ,tr.rotationY(np.pi/2)])
    roof1Node.childs += [roof]


    roof2Node = sg.SceneGraphNode('roof1')
    roof2Node.transform = tr.matmul([tr.scale(1,1,1.15),tr.translate(-1/3,2/3,0), tr.rotationZ(-np.pi/4) ,tr.rotationY(np.pi/2)])
    roof2Node.childs += [roof]


    scene = sg.SceneGraphNode('system')
    scene.transform = tr.uniformScale(1.2)
    scene.childs += [wall1Node]
    scene.childs += [wall2Node]
    scene.childs += [wall3Node]
    scene.childs += [wall4Node]
    scene.childs += [wall5Node]
    scene.childs += [wall6Node]
    scene.childs += [roof1Node]
    scene.childs += [roof2Node]
    
    return scene

# TAREA3: Implementa la función "createWall" que crea un objeto que representa un muro
# y devuelve un nodo de un grafo de escena (un objeto sg.SceneGraphNode) que representa toda la geometría y las texturas
# Esta función recibe como parámetro el pipeline que se usa para las texturas (texPipeline)

def createWall(pipeline):
    wall = createGPUShape(pipeline, bs.createTextureQuad(30,4))
    wall.texture = es.textureSimpleSetup( # se settean las texturas de la pared
        getAssetPath("wall2.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)


    wallNode = sg.SceneGraphNode('wall') # y se colocan en las dimensiones que queremos
    wallNode.transform = tr.matmul([tr.rotationY(np.pi/2), tr.scale(7.5,.5,7.5)])
    wallNode.childs += [wall]

    scene = sg.SceneGraphNode('system')
    scene.childs += [wallNode]

    return scene

# TAREA3: Esta función crea un grafo de escena especial para el auto.
def createCarScene(pipeline):
    chasis = createOFFShape(pipeline, 'alfa2.off', 1.0, 0.0, 0.0)
    wheel = createOFFShape(pipeline, 'wheel.off', 0.0, 0.0, 0.0)

    scale = 2.0
    rotatingWheelNode = sg.SceneGraphNode('rotatingWheel')
    rotatingWheelNode.childs += [wheel]

    chasisNode = sg.SceneGraphNode('chasis')
    chasisNode.transform = tr.uniformScale(scale)
    chasisNode.childs += [chasis]

    wheel1Node = sg.SceneGraphNode('wheel1')
    wheel1Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.056390,0.037409,0.091705)])
    wheel1Node.childs += [rotatingWheelNode]

    wheel2Node = sg.SceneGraphNode('wheel2')
    wheel2Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.060390,0.037409,-0.091705)])
    wheel2Node.childs += [rotatingWheelNode]

    wheel3Node = sg.SceneGraphNode('wheel3')
    wheel3Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(-0.056390,0.037409,0.091705)])
    wheel3Node.childs += [rotatingWheelNode]

    wheel4Node = sg.SceneGraphNode('wheel4')
    wheel4Node.transform = tr.matmul([tr.uniformScale(scale),tr.translate(0.066090,0.037409,-0.091705)])
    wheel4Node.childs += [rotatingWheelNode]

    car1 = sg.SceneGraphNode('car1')
    car1.transform = tr.matmul([tr.translate(2.0, -0.037409, 5.0),tr.rotationY(np.pi)])
    car1.childs += [chasisNode]
    car1.childs += [wheel1Node]
    car1.childs += [wheel2Node]
    car1.childs += [wheel3Node]
    car1.childs += [wheel4Node]

    scene = sg.SceneGraphNode('system')
    scene.childs += [car1]

    return scene

# TAREA3: Esta función crea toda la escena estática y texturada de esta aplicación.
# Por ahora ya están implementadas: la pista y el terreno
# En esta función debes incorporar las casas y muros alrededor de la pista

def createStaticScene(pipeline):

    roadBaseShape = createGPUShape(pipeline, bs.createTextureQuad(1.0, 1.0))
    roadBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Road_001_basecolor.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    sandBaseShape = createGPUShape(pipeline, createTiledFloor(50))
    sandBaseShape.texture = es.textureSimpleSetup(
        getAssetPath("Sand 002_COLOR.jpg"), GL_REPEAT, GL_REPEAT, GL_LINEAR_MIPMAP_LINEAR, GL_NEAREST)
    glGenerateMipmap(GL_TEXTURE_2D)

    arcShape = createGPUShape(pipeline, createTexturedArc(1.5))
    arcShape.texture = roadBaseShape.texture

    
    roadBaseNode = sg.SceneGraphNode('plane')
    roadBaseNode.transform = tr.rotationX(-np.pi/2)
    roadBaseNode.childs += [roadBaseShape]

    arcNode = sg.SceneGraphNode('arc')
    arcNode.childs += [arcShape]

    sandNode = sg.SceneGraphNode('sand')
    sandNode.transform = tr.translate(0.0,-0.1,0.0)
    sandNode.childs += [sandBaseShape]

    linearSector = sg.SceneGraphNode('linearSector')
        
    for i in range(10):
        node = sg.SceneGraphNode('road'+str(i)+'_ls')
        node.transform = tr.translate(0.0,0.0,-1.0*i)
        node.childs += [roadBaseNode]
        linearSector.childs += [node]

    linearSectorLeft = sg.SceneGraphNode('lsLeft')
    linearSectorLeft.transform = tr.translate(-2.0, 0.0, 5.0)
    linearSectorLeft.childs += [linearSector]

    linearSectorRight = sg.SceneGraphNode('lsRight')
    linearSectorRight.transform = tr.translate(2.0, 0.0, 5.0)
    linearSectorRight.childs += [linearSector]

    arcTop = sg.SceneGraphNode('arcTop')
    arcTop.transform = tr.translate(0.0,0.0,-4.5)
    arcTop.childs += [arcNode]

    arcBottom = sg.SceneGraphNode('arcBottom')
    arcBottom.transform = tr.matmul([tr.translate(0.0,0.0,5.5), tr.rotationY(np.pi)])
    arcBottom.childs += [arcNode]

    # aqui se añaden las casas y las paredes a la escena

    house1 = createHouse(pipeline)

    house2 = createHouse(pipeline)
    house2.transform = tr.matmul([tr.translate(4,0,0), tr.rotationY(np.pi/2)])

    house3 = createHouse(pipeline)
    house3.transform = tr.translate(0,0,10)

    house4 = createHouse(pipeline)
    house4.transform = tr.translate(0,0,-10)

    house5 = createHouse(pipeline)
    house5.transform = tr.matmul([tr.translate(4.2,0,4), tr.rotationY(np.pi/2)])

    house6 = createHouse(pipeline)
    house6.transform = tr.matmul([tr.translate(4.2,0,-4), tr.rotationY(np.pi/2)])

    houses = sg.SceneGraphNode('houses')
    houses.transform = tr.translate(-8.2,0,0)
    houses.childs += [house2,house5,house6]


    wall1 = createWall(pipeline)
    wall1.transform = tr.translate(2.55,0,0)

    wall2 = createWall(pipeline)
    wall2.transform = tr.translate(1.45,0,0) 

    walls = sg.SceneGraphNode('walls')
    walls.transform = tr.translate(-4,0,0)
    walls.childs += [wall1,wall2]
    
    scene = sg.SceneGraphNode('system')
    scene.childs += [linearSectorLeft]
    scene.childs += [linearSectorRight]
    scene.childs += [arcTop]
    scene.childs += [arcBottom]
    scene.childs += [sandNode]
    scene.childs += [house1]
    scene.childs += [house2]
    scene.childs += [house3]
    scene.childs += [house4]
    scene.childs += [house5]
    scene.childs += [house6]
    scene.childs += [houses]
    scene.childs += [wall1]
    scene.childs += [wall2]
    scene.childs += [walls]
    
    return scene

if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 2"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    axisPipeline = es.SimpleModelViewProjectionShaderProgram()
    texPipeline = es.SimpleTextureModelViewProjectionShaderProgram()
    lightPipeline = ls.SimpleGouraudShaderProgram()
    
    # Telling OpenGL to use our shader program
    glUseProgram(axisPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    axisPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    #NOTA: Aqui creas un objeto con tu escena
    dibujo = createStaticScene(texPipeline)
    car =createCarScene(lightPipeline)

    setPlot(texPipeline, axisPipeline,lightPipeline)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    while not glfw.window_should_close(window):

        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))

        # Using GLFW to check for input events
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Acá se generan el movimiento creado por el presionar de las teclas por cada iteracion

        if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
            # con W, el auto avanza la cantidad speed hacia la direccion en la que apunta (adelante del auto)
            car.transform = tr.matmul([tr.translate(controller.dir[0]*controller.speed,0.0,controller.dir[2]*controller.speed), car.transform]) 
            # se actualiza la posicion del auto
            controller.car += np.array([controller.dir[0]*controller.speed,0.0,controller.dir[2]*controller.speed])
            

        if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
            # con S el auto retrocede una cantidad speed
            car.transform = tr.matmul([tr.translate(-controller.dir[0]*controller.speed,0.0,-controller.dir[2]*controller.speed), car.transform]) 
            controller.car -= np.array([controller.dir[0]*controller.speed,0.0,controller.dir[2]*controller.speed])


        if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
            # con A, se rota el auto en su propio eje hacia la izquierda
            car.transform = tr.matmul([car.transform, tr.translate(2.0,0,5.0),tr.rotationY(controller.rotSpeed), tr.translate(-2.0,0,-5.0)])
            # y cambia el vector direccion, para que avance ahora hacia donde apunta el auto nuevamente
            controller.dir = np.array([controller.dir[0]*np.cos(controller.rotSpeed) + controller.dir[2]*np.sin(controller.rotSpeed),0, controller.dir[2]*np.cos(controller.rotSpeed) - controller.dir[0]*np.sin(controller.rotSpeed)])



        if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
            # idem que A, pero hacia la derecha
            car.transform = tr.matmul([car.transform, tr.translate(2.0,0,5.0),tr.rotationY(-controller.rotSpeed), tr.translate(-2.0,0,-5.0)])
            controller.dir = np.array([controller.dir[0]*np.cos(-controller.rotSpeed) + controller.dir[2]*np.sin(-controller.rotSpeed),0, controller.dir[2]*np.cos(-controller.rotSpeed) - controller.dir[0]*np.sin(-controller.rotSpeed)])

        setView(texPipeline, axisPipeline, lightPipeline)

        if controller.showAxis:
            glUseProgram(axisPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(axisPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            axisPipeline.drawCall(gpuAxis, GL_LINES)

        #NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(texPipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, texPipeline, "model")

        glUseProgram(lightPipeline.shaderProgram)
        sg.drawSceneGraphNode(car, lightPipeline, "model")

        

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()
    

    glfw.terminate()