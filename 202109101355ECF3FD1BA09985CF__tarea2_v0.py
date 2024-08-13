# coding=utf-8
"""Tarea 2, by Matías Sothers, modelo moto 5"""

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

__author__ = "Ivan Sipiran"
__license__ = "MIT"


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True
        self.viewPos = np.array([10, 10, 10])
        self.camUp = np.array([0, 1, 0])
        self.distance = 10


controller = Controller()


def setPlot(pipeline, mvpPipeline):
    projection = tr.perspective(45, float(width) / float(height), 0.1, 100)

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 5, 5, 5)

    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 1000)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)


def setView(pipeline, mvpPipeline):
    view = tr.lookAt(
        controller.viewPos,
        np.array([0, 0, 0]),
        controller.camUp
    )

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), controller.viewPos[0],
                controller.viewPos[1], controller.viewPos[2])


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

    elif key == glfw.KEY_1:
        controller.viewPos = np.array(
            [controller.distance, controller.distance, controller.distance])  # Vista diagonal 1
        controller.camUp = np.array([0, 1, 0])

    elif key == glfw.KEY_2:
        controller.viewPos = np.array([0, 0, controller.distance])  # Vista frontal
        controller.camUp = np.array([0, 1, 0])

    elif key == glfw.KEY_3:
        controller.viewPos = np.array([controller.distance, 0, controller.distance])  # Vista lateral
        controller.camUp = np.array([0, 1, 0])

    elif key == glfw.KEY_4:
        controller.viewPos = np.array([0, controller.distance, 0])  # Vista superior
        controller.camUp = np.array([1, 0, 0])

    elif key == glfw.KEY_5:
        controller.viewPos = np.array(
            [controller.distance, controller.distance, -controller.distance])  # Vista diagonal 2
        controller.camUp = np.array([0, 1, 0])

    elif key == glfw.KEY_6:
        controller.viewPos = np.array(
            [-controller.distance, controller.distance, -controller.distance])  # Vista diagonal 2
        controller.camUp = np.array([0, 1, 0])

    elif key == glfw.KEY_7:
        controller.viewPos = np.array(
            [-controller.distance, controller.distance, controller.distance])  # Vista diagonal 2
        controller.camUp = np.array([0, 1, 0])


    else:
        print('Unknown key')


def createGPUShape(pipeline, shape):
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape


# NOTA: Aqui creas tu escena. En escencia, sólo tendrías que modificar esta función.
def createScene(pipeline):
    cube = createGPUShape(pipeline, bs.createColorCubeTarea2(0.75, 0.5, 0.0))
    cube_w = createGPUShape(pipeline, bs.createColorCubeTarea2(0.95, 0.94, 0.96))
    cylinder = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.99, 0.87, 0.0))
    cylinder_k = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.0, 0.0, 0.0))
    cylinder2 = createGPUShape(pipeline, bs.createColorCylinderTarea2(0.75, 0.5, 0.03))


    ## Cuerpo de la guitarra
    cuerpo1Node = sg.SceneGraphNode('cuerpo1')
    cuerpo1Node.transform = tr.matmul([tr.translate(-.5, 0, 0), tr.scale(.9, 0.33, 1)])
    cuerpo1Node.childs += [cylinder]

    cuerpo2Node = sg.SceneGraphNode('cuerpo2')
    cuerpo2Node.transform = tr.matmul([tr.translate(.5, 0, 0), tr.scale(.7, .33, .8)])
    cuerpo2Node.childs += [cylinder]

    holeNode = sg.SceneGraphNode('hole')
    holeNode.transform = tr.matmul([tr.translate(0.4, .33, 0), tr.scale(.35, 0.01, .35)])
    holeNode.childs += [cylinder_k]

    cuerpoNode = sg.SceneGraphNode('cuerpo')
    cuerpoNode.childs += [cuerpo2Node, cuerpo1Node, holeNode]

    ## Puente de la guitarra

    puente1Node = sg.SceneGraphNode('puente1')
    puente1Node.transform = tr.matmul([tr.translate(0, 0.33, 0), tr.scale(1.2, 0.05, 0.15)])
    puente1Node.childs += [cube]

    puente2Node = sg.SceneGraphNode('puente2')
    puente2Node.transform = tr.matmul([tr.translate(-.66, 0, 0), tr.scale(.066, .33, .15)])
    puente2Node.childs += [cube]

    ## Clavijero (parte de arriba del puente)
    clavijero1Node = sg.SceneGraphNode('clavijero1')
    clavijero1Node.transform = tr.matmul([tr.translate(0, 0, .15/5), tr.shearing(0, 0, 0, 0.5/5, 0, 0), tr.scale(.25, 1, 1)])
    clavijero1Node.childs += [puente1Node]

    clavijero2Node = sg.SceneGraphNode('clavijero2')
    clavijero2Node.transform = tr.matmul([tr.translate(0, 0, -.15/5), tr.shearing(0, 0, 0, -0.5/5, 0, 0), tr.scale(.25, 1, 1)])
    clavijero2Node.childs += [puente1Node]

    clavijeroNode = sg.SceneGraphNode('clavijero')
    clavijeroNode.transform = tr.translate(1.53,0,0)
    clavijeroNode.childs += [clavijero1Node, clavijero2Node]

    ## Cada clavija
    preClavijaNode = sg.SceneGraphNode('preClavija')
    preClavijaNode.transform = tr.matmul([tr.rotationX(np.pi/2), tr.scale(.01,.1,.01)])
    preClavijaNode.childs += [cylinder2]

    perillaNode = sg.SceneGraphNode('perilla')
    perillaNode.transform = tr.matmul([tr.translate(0, 0, -.05), tr.scale(.07, .01, .05)])
    perillaNode.childs += [cube_w]

    clavijaNode = sg.SceneGraphNode('clavija')
    clavijaNode.transform = tr.uniformScale(.9)
    clavijaNode.childs += [perillaNode, preClavijaNode]

    ## Posicionamos las clavijas
    clavija1Node = sg.SceneGraphNode('clavija1')
    clavija1Node.transform = tr.translate(.233, 0.33, -.23)
    clavija1Node.childs += [clavijaNode]

    clavija2Node = sg.SceneGraphNode('clavija2')
    clavija2Node.transform = tr.translate(0.05, 0.33, -.23)
    clavija2Node.childs += [clavijaNode]

    clavija3Node = sg.SceneGraphNode('clavija3')
    clavija3Node.transform = tr.translate(-0.133, 0.33, -.23)
    clavija3Node.childs += [clavijaNode]

    clavijasLeftNode = sg.SceneGraphNode('clavijasLeft')
    clavijasLeftNode.childs += [clavija1Node, clavija2Node, clavija3Node]

    clavijasRightNode = sg.SceneGraphNode('clavijasRight')
    clavijasRightNode.transform = tr.matmul([tr.translate(0, 0.66, 0), tr.rotationX(np.pi)])
    clavijasRightNode.childs += [clavijasLeftNode]


    ## Y las agregamos al clavijero
    clavijeroNode.childs += [clavijasRightNode, clavijasLeftNode]

    ## Juntamos las partes del puente
    puenteNode = sg.SceneGraphNode('puente')
    puenteNode.transform = tr.translate(1.9, 0, 0)
    puenteNode.childs += [puente1Node, puente2Node, clavijeroNode]

    ## Juntamos el puente con el cuerpo
    guitarraNode = sg.SceneGraphNode('guitarra')
    guitarraNode.transform = tr.translate(-.33, 0, 0)
    guitarraNode.childs += [cuerpoNode, puenteNode]


    scene = sg.SceneGraphNode('system')
    scene.childs += [guitarraNode]

    return scene


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 800
    height = 800
    title = "Tarea 2: Modelo moto 5"
    window = glfw.create_window(width, height, title, None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimpleGouraudShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    mvpPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)

    # NOTA: Aqui creas un objeto con tu escena
    dibujo = createScene(pipeline)

    setPlot(pipeline, mvpPipeline)

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

        setView(pipeline, mvpPipeline)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawCall(gpuAxis, GL_LINES)

        # NOTA: Aquí dibujas tu objeto de escena
        glUseProgram(pipeline.shaderProgram)
        sg.drawSceneGraphNode(dibujo, pipeline, "model")

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # freeing GPU memory
    gpuAxis.clear()
    dibujo.clear()

    glfw.terminate()
