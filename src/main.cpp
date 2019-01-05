// This example is heavily based on the tutorial at https://open.gl

// OpenGL Helpers to reduce the clutter
#include "Helpers.h"

// GLFW is necessary to handle the OpenGL context
#include <GLFW/glfw3.h>

// Linear Algebra Library
#include <Eigen/Core>
#include <Eigen/Geometry>

// Timer
#include <chrono>

#include <iostream>
#include <fstream>
#include <string>
#include <list>
#include <cstdlib>
using namespace std;

#define PI 3.14159265

 Program program;

// VertexBufferObject wrapper
VertexBufferObject VBO;
//VertexBufferObject CBO;
VertexBufferObject NBO;
IndexBufferObject IBO;

// Contains the vertex positions
Eigen::MatrixXf V;
Eigen::MatrixXf C;
Eigen::MatrixXf N;
Eigen::MatrixXf B;
vector<unsigned int> I;

Eigen::MatrixXf colorCodes;

int screen_width = 640;
int screen_height = 480;

bool first_load = true;

enum RenderType
{
    WIRE_FRAME,
    FLAT_SHADING,
    PHONG_SHADING
};

RenderType rendering;

enum ObjectName
{
    UNIT_CUBE,
    BUNNY,
    BUMPY_CUBE
};

class Object {
public:
    unsigned int id;
    ObjectName name;
    unsigned int indexSize;
    unsigned int indexOffset;
    unsigned int vertexColSize;
    unsigned int vertexOffset;
    Eigen::Vector3f center;
    Eigen::Vector3f baryCenter;
    
    Object(){};
    
    Object(unsigned int id, ObjectName name, unsigned int indexSize, unsigned int indexOffset, unsigned int vertexColSize, unsigned int vertexOffset, Eigen::Vector3f center){
        this->id = id;
        this->name = name;
        this->indexSize = indexSize;
        this->indexOffset = indexOffset;
        this->vertexColSize = vertexColSize;
        this->vertexOffset = vertexOffset;
        this->center = center;
    };
};

class Instance {
public:
    unsigned int id;
    Object object;
    Eigen::Matrix4f baseMVP;
    Eigen::Matrix4f transformationMVP;
    Eigen::Matrix4f baseModel;
    Eigen::Matrix4f transformationModel;
    Eigen::Vector3f color;
    
    Instance(unsigned int id, Object object, Eigen::Matrix4f baseMVP, Eigen::Matrix4f baseModel, Eigen::Vector3f color){
        this->id = id;
        this->object = object;
        this->baseMVP = baseMVP;
        this->transformationMVP = Eigen::Matrix4f::Identity();
        this->baseModel = baseModel;
        this->transformationModel = Eigen::Matrix4f::Identity();
        this->color = color;
    };

    Eigen::Matrix4f getMVP() {
        return this->transformationMVP * this->baseMVP;
    };
    
    Eigen::Matrix4f getModel() {
        return this->transformationModel * this->baseModel;
    };
};

namespace LightSource {
    Eigen::Vector3f position = Eigen::Vector3f(0.0, 1.0, 2.0);
    Eigen::Vector3f color = Eigen::Vector3f(1.0, 1.0, 1.0);
};

Eigen::Vector3f cameraPosition(-1.0, 1.0, 2.0);
Eigen::Vector3f target(0.0, 0.0, 0.0);
Eigen::Vector3f worldUp(0.0, 1.0, 0.0);

list<Object> objectCollection;
list<Instance> instanceCollection;

enum Action
{
    INSERTION,
    TRANSLATION,
    DELETION
};

Action actionTriggered;

enum Transformation
{
    COLOR,
    SCALE,
    TRANSLATE,
    ROTATE
};

enum Projection{
    Perspective,
    Orthographic
};

Projection projectionType;

int selectedInstanceId = -1;
bool setTotalView = false;
Eigen::Matrix4f totalView(4,4);
int no_of_clicks_translate = 0;
bool enableCursorTrack = false;
bool colorUpdated = false;

bool ptInTriangle(float px, float py, float v0x, float v0y, float v1x, float v1y, float v2x, float v2y) {
    float dX = px-v2x;
    float dY = py-v2y;
    float dX21 = v2x-v1x;
    float dY12 = v1y-v2y;
    float D = dY12*(v0x-v2x) + dX21*(v0y-v2y);
    float s = dY12*dX + dX21*dY;
    float t = (v2y-v0y)*dX + (v0x-v2x)*dY;
    if (D<0) return s<=0 && t<=0 && s+t>=D;
    return s>=0 && t>=0 && s+t<=D;
}

Eigen::Vector3f centroid_of_triangle(Eigen::Vector3f v0, Eigen::Vector3f v1, Eigen::Vector3f v2){
    Eigen::Vector3f centroid;
    centroid(0) = (v0.x() + v1.x() + v2.x()) / 3;
    centroid(1) = (v0.y() + v1.y() + v2.y()) / 3;
    centroid(2) = (v0.z() + v1.z() + v2.z()) / 3;
    return centroid;
}

// Custom implementation of the LookAt function
Eigen::Matrix4f calculate_lookAt_matrix(Eigen::Vector3f position, Eigen::Vector3f target, Eigen::Vector3f worldUp)
{
    // 1. Position = known
    // 2. Calculate cameraDirection
    Eigen::Vector3f zaxis = (position - target).normalized();
    // 3. Get positive right axis vector
    Eigen::Vector3f worldUp_normalized = worldUp.normalized();
    Eigen::Vector3f xaxis = (worldUp_normalized.cross(zaxis)).normalized();
    // 4. Calculate camera up vector
    Eigen::Vector3f yaxis = zaxis.cross(xaxis);
    
    // Create translation and rotation matrix
    // In glm we access elements as mat[col][row] due to column-major layout
    Eigen::Matrix4f  translation = Eigen::Matrix4f::Identity(); // Identity matrix by default
    translation(0, 3) = -position.x();
    translation(1, 3) = -position.y();
    translation(2, 3) = -position.z();
    Eigen::Matrix4f  rotation = Eigen::Matrix4f::Identity();
    rotation(0, 0) = xaxis.x();
    rotation(0, 1) = xaxis.y();
    rotation(0, 2) = xaxis.z();
    rotation(1, 0) = yaxis.x();
    rotation(1, 1) = yaxis.y();
    rotation(1, 2) = yaxis.z();
    rotation(2, 0) = zaxis.x();
    rotation(2, 1) = zaxis.y();
    rotation(2, 2) = zaxis.z();
    
    // Return lookAt matrix as combination of translation and rotation matrix
    return rotation * translation; // Remember to read from right to left (first translation then rotation)
}

Eigen::Matrix4f perspective(float degree, float aspect,float zNear,float zFar)
{
    float tanHalfFovy = tan((degree*PI/180) /2);
    
    Eigen::Matrix4f result = Eigen::Matrix4f::Zero();
    result(0, 0) = 1/(aspect * tanHalfFovy);
    result(1, 1) = 1/(tanHalfFovy);
    result(2, 2) = -(zFar + zNear) / (zFar - zNear);
    result(3, 2) = -1;
    result(2, 3) = - (2 * zFar * zNear) / (zFar - zNear);
    return result;
}

Eigen::Matrix4f ortho(float xmin, float xmax, float ymin, float ymax, float zmin, float zmax){
    Eigen::Matrix4f result = Eigen::Matrix4f::Identity();
    result(0, 0) = 2 / (xmax - xmin);
    result(1, 1) = 2 / (ymax - ymin);
    result(2, 2) = -2 / (zmax - zmin);
    result(0, 3) = -(xmax + xmin) / (xmax - xmin);
    result(1, 3) = -(ymax + ymin) / (ymax - ymin);
    result(2, 3) = -(zmax + zmin) / (zmax - zmin);
    
    return result;
}

Eigen::Matrix4f scale(float zoom){
    // Contains the rotation transformation
    Eigen::Matrix4f scale(4,4);
    
    scale <<
    zoom, 0, 0, 0,
    0, zoom, 0, 0,
    0, 0, zoom, 0,
    0, 0, 0, 1;
    
    return scale;
}

Eigen::Matrix4f rotationAboutX(double degree){
    // Contains the rotation transformation
    Eigen::Matrix4f rotation(4,4);
    
    rotation <<
    1, 0, 0, 0,
    0, cos(degree*PI/180), -sin(degree*PI/180), 0,
    0, sin(degree*PI/180), cos(degree*PI/180), 0,
    0, 0, 0, 1;
    
    return rotation;
}

Eigen::Matrix4f rotationAboutY(double degree){
    // Contains the rotation transformation
    Eigen::Matrix4f rotation(4,4);
    
    rotation <<
    cos(degree*PI/180), 0, -sin(degree*PI/180), 0,
    0, 1, 0, 0,
    -sin(degree*PI/180), 0, cos(degree*PI/180), 0,
    0, 0, 0, 1;
    
    return rotation;
}

Eigen::Matrix4f rotationAboutZ(double degree){
    // Contains the rotation transformation
    Eigen::Matrix4f rotation(4,4);
    
    rotation <<
    cos(degree*PI/180), -sin(degree*PI/180), 0, 0,
    sin(degree*PI/180), cos(degree*PI/180), 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1;
    
    return rotation;
}

Eigen::Matrix4f translate(float x, float y, float z){
    // Contains the translation transformation
    Eigen::Matrix4f translate(4,4);
    
    translate <<
    1, 0, 0, x,
    0, 1, 0, y,
    0, 0, 1, z,
    0, 0, 0, 1;
    
    return translate;
}

Eigen::Matrix4f translate(Eigen::Vector3f translate_by){
    // Contains the translation transformation
    Eigen::Matrix4f translate(4,4);
    
    translate <<
    1, 0, 0, translate_by.x(),
    0, 1, 0, translate_by.y(),
    0, 0, 1, translate_by.z(),
    0, 0, 0, 1;
    
    return translate;
}

void updateColorToTheSelectedInstance(int colorCodeIndex) {
    if(selectedInstanceId > 0){
        for(auto& instance: instanceCollection){
            if(instance.id == selectedInstanceId){
                instance.color = colorCodes.col(colorCodeIndex);
                colorUpdated = true;
                break;
            }
        }
    }
}

bool loadMeshFromFile(string filename, Eigen::Vector3f &objectCenter) {
    
    ifstream file;
    int no_of_vertices;
    int no_of_faces;
    int previous_V_col_size = V.cols();
    int previous_I_size = I.size();
    int i = 0;
    int totalIndicesPerRow = 0;
    Eigen::Vector3f vertexSum = Eigen::Vector3f::Zero();
    
    file.open ("../../data/"+filename);
    if (!file.is_open()) {
        cout << "Try checking file presence one level above in directory" << endl;
        file.open ("../data/"+filename);
        if(!file.is_open()){
            cout << "Oops! no file found!" << endl;
            return false;
        }
    }
    
    std::string eachLine;
    int lineNumber = 0;
    // Get each line
    while (std::getline (file, eachLine)) {
        // Use std::stringstream to isolate words using operator >>
        std::stringstream wordStream (eachLine);
        
        std::string eachWord;
        int wordNumber = 0;
        
        if(lineNumber == 0) {
            while (wordStream >> eachWord) {
                if(eachWord != "OFF") {
                    std::cout << "Not a OFF file" << endl;
                    return false;
                }
                wordNumber++;
            }
        } else if(lineNumber == 1){
            while (wordStream >> eachWord) {
                if(wordNumber == 0){
                    no_of_vertices = stoi(eachWord);
                } else if(wordNumber == 1) {
                    no_of_faces = stoi(eachWord);
                }
                wordNumber++;
            }
            V.conservativeResize(3, previous_V_col_size + no_of_vertices);
            //C.conservativeResize(3, previous_V_col_size + no_of_vertices);
        } else if(lineNumber > 1 && lineNumber < no_of_vertices + 2){
            while (wordStream >> eachWord) {
                if (wordNumber < 3) {
                    V(wordNumber, previous_V_col_size + lineNumber -2) = stof(eachWord);
                }
                if(wordNumber == 2) {
                    vertexSum = vertexSum + V.col(previous_V_col_size + lineNumber -2);
                }
                wordNumber++;
            }
        } else {
            while (wordStream >> eachWord) {
                if(lineNumber == no_of_vertices + 2 && wordNumber == 0){
                    totalIndicesPerRow = stoi(eachWord);
                }
                if (wordNumber > 0 && wordNumber < totalIndicesPerRow + 1) {
                    I.push_back(previous_V_col_size + stoi(eachWord));
                }
                wordNumber++;
            }
        }
        lineNumber++;
    }
    objectCenter = vertexSum / no_of_vertices;
    file.close ();
    return true;
}

void drawOutput()
{
    if(I.size() != 0){
        GLenum mode = rendering == RenderType::WIRE_FRAME ? GL_LINE_LOOP : GL_TRIANGLES;
       for (auto& instance : instanceCollection) {
           //set Stencil value
            glStencilFunc(GL_ALWAYS, instance.id, -1);
            // in the vertex shader
            glUniformMatrix4fv(program.uniform("mvp"), 1, GL_FALSE, instance.getMVP().data());
            glUniformMatrix4fv(program.uniform("model"), 1, GL_FALSE, instance.getModel().data());
            if(selectedInstanceId == instance.id && !colorUpdated){
               glUniform3fv(program.uniform("objectColor"), 1, colorCodes.col(12).data());
            } else {
                glUniform3fv(program.uniform("objectColor"), 1, instance.color.data());
            }
           
    
           /* if(instance.object.name == ObjectName::UNIT_CUBE){
                glDrawElements(mode, instance.object.indexSize, GL_UNSIGNED_INT, (unsigned int *) instance.object.indexOffset);
            } else {
                glDrawElements(mode, instance.object.indexSize, GL_UNSIGNED_INT, (unsigned int *) instance.object.indexOffset);
            } */
            glDrawElements(mode, instance.object.indexSize, GL_UNSIGNED_INT, (void *) instance.object.indexOffset);
           /*if(rendering == RenderType::FLAT_SHADING){
               glUniform3f(program.uniform("objectColor"), 0.5, 0.5, 0.5);
               glDrawElements(GL_LINE_LOOP, instance.object.indexSize, GL_UNSIGNED_INT, (void *) instance.object.indexOffset);
           }*/
        }
    }
}

Eigen::Matrix4f calculateBaseModel(Object object){
    float objectScale;
    Eigen::Matrix4f baseModel;
    
    switch (object.name) {
        case ObjectName::UNIT_CUBE:
            objectScale = 0.2;
            break;
        case ObjectName::BUMPY_CUBE:
            objectScale = 0.07;
            break;
        case ObjectName::BUNNY:
            objectScale = 2.5;
            break;
        default:
            break;
    }
    
    float rand_x = (rand() % 151)/100.0 -0.75; // x values btw -0.75, 0.75
    float rand_y = (rand() % 151)/100.0 -0.75; // y values btw -0.75, 0.75
    //float rand_z = (rand() % 3) -4; // z values btw -2, -4
    
    baseModel = translate(rand_x, rand_y, 0) * translate(target - object.center) * scale(objectScale);
    return baseModel;
}

Eigen::Matrix4f calculateBaseMVP(Eigen::Matrix4f baseModel){
    Eigen::Matrix4f View;
    Eigen::Matrix4f Projection;
    Eigen::Matrix4f baseMvp;
    View = calculate_lookAt_matrix(cameraPosition, target, worldUp);
    float aspectRatio = 1.0f * screen_width / screen_height;
    if(projectionType == Projection::Orthographic){
       Projection = ortho(0.0f, 2.0f, 0.0f, 2.0f*(1.0f/aspectRatio), 0.1f, 10.0f) * translate(1.0, 1.0/aspectRatio, 0.0);
    } else {
       Projection = perspective(45.0, aspectRatio, 0.1f, 10.0f);
    }
    baseMvp = Projection * View * baseModel;
    return baseMvp;
}

void computeNormalsAndBarycenter(Object& object){
    N.conservativeResize(V.rows(), V.cols());
    B.conservativeResize(V.rows(), V.cols());
    Eigen::VectorXf track_no_shared_faces_per_vertex;
    track_no_shared_faces_per_vertex.conservativeResize(V.cols());
    unsigned int indexSize = object.indexSize;
    unsigned int indexOffset = object.indexOffset;
    unsigned int vertexColSize = object.vertexColSize;
    unsigned int vertexOffset = object.vertexOffset;
    
    for(int i = vertexOffset; i < vertexColSize ; i++){
        N.col(i) << 0.0, 0.0, 0.0;
        B.col(i) << 0.0, 0.0, 0.0;
        track_no_shared_faces_per_vertex(i) = 0;
    }
        
        for( int i=indexOffset; i < indexSize;)
        {
            
            Eigen::Vector3f V0 = V.col(I[i]);
            Eigen::Vector3f V1 = V.col(I[i+1]);
            Eigen::Vector3f V2 = V.col(I[i+2]);
            
            Eigen::Vector3f normal = (V1-V0).cross(V2-V0).normalized();
            
            N.col(I[i]) += normal;
            track_no_shared_faces_per_vertex(I[i]) += 1;
            N.col(I[i+1]) += normal;
            track_no_shared_faces_per_vertex(I[i+1]) += 1;
            N.col(I[i+2]) += normal;
            track_no_shared_faces_per_vertex(I[i+2]) += 1;
            
            //Eigen::Vector3f baryCenter = centroid_of_triangle(V0, V1, V2);
            
            i = i +3;
        }
    
    for(int i = vertexOffset; i < vertexColSize ; i++){
        N.col(i) = (N.col(i)/track_no_shared_faces_per_vertex(I[i])).normalized(); // normalize the average of the normal of all shared faces for a vertex
    }
    NBO.update(N);
}

void addObjectToTheScene(ObjectName objectName){
    bool object_loaded = false;
    for(auto const & object: objectCollection){
        if(object.name == objectName){
            object_loaded = true;
        }
    }
    if(!object_loaded){
        unsigned int previousObjectIndexSize = I.size();
        unsigned int previousObjectvertexColSize = V.cols();
        string filename;
        switch (objectName) {
            case ObjectName::UNIT_CUBE:
                filename = "unit_cube_TRIANGLES.off";
                break;
            case ObjectName::BUMPY_CUBE:
                filename = "bumpy_cube.off";
                break;
            case ObjectName::BUNNY:
                filename = "bunny.off";
                break;
            default:
                break;
        }
        Eigen::Vector3f objectCenter;
        if(loadMeshFromFile(filename, objectCenter)) {
            Object newObject = Object(objectCollection.size()+1 , objectName, I.size() - previousObjectIndexSize, previousObjectIndexSize, V.cols() - previousObjectvertexColSize, previousObjectvertexColSize, objectCenter);
            objectCollection.push_back(newObject);
            computeNormalsAndBarycenter(newObject);
            
            VBO.update(V);
            IBO.update(I);
            
            if(first_load){
                program.bindVertexAttribArray("position", VBO);
                program.bindVertexAttribArray("normal", NBO);
                //program.bindVertexAttribArray("color", CBO);
                first_load = false;
            }
        }
    }
    
    Eigen::Vector3f color;
    Eigen::Matrix4f mvp;
    switch (objectName) {
        case ObjectName::UNIT_CUBE:
            color << colorCodes.col(9);
            break;
        case ObjectName::BUMPY_CUBE:
            color << colorCodes.col(10);
            break;
        case ObjectName::BUNNY:
            color << colorCodes.col(11);
            break;
        default:
            break;
    }
    
    for(auto const& object: objectCollection){
        if(object.name == objectName){
            Eigen::Matrix4f baseModel = calculateBaseModel(object);
            Eigen::Matrix4f baseMVP = calculateBaseMVP(baseModel);
            Instance instance = Instance(instanceCollection.size()+1, object, baseMVP, baseModel, color);
            instanceCollection.push_back(instance);
        }
    }
}

void updateChangesToSelectedInstance(){
    
}

float pointer_x = 0.0;
float pointer_y = 0.0;
Eigen::Matrix4f beforeMVP = Eigen::Matrix4f::Identity();

void cursor_position_callback(GLFWwindow *window, double x, double y)
{
    if (enableCursorTrack)
    {
        // Get the size of the window
        int width, height;
        glfwGetWindowSize(window, &width, &height);
        
        // Convert screen position to world coordinates
        Eigen::Vector4f p_screen(x,height-1-y,0,1); // NOTE: y axis is flipped in glfw
        Eigen::Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
       
        for(auto& instance: instanceCollection){
            if(instance.id == selectedInstanceId){
                Eigen::Vector4f p_world = setTotalView && !totalView.isZero() ? totalView.inverse() * p_canonical : p_canonical;
                float translation_x = p_world.x() - pointer_x;
                float translation_y  = p_world.y() - pointer_y;
                Eigen::Matrix4f transformMatrix = translate(translation_x, translation_y, 0.0);
                instance.transformationMVP = transformMatrix  * instance.transformationMVP;
                instance.transformationModel = transformMatrix * instance.transformationModel;
                pointer_x = p_world.x();
                pointer_y = p_world.y();
            }
        }
    }
}

/*void findModelSelected(float xworld, float yworld) {
    for(auto const& instance: instanceCollection){
        for(unsigned int i = instance.object.indexOffset; i < instance.object.indexSize;){
            Eigen::Vector4f transformedVertex1 = instance.mvp * Eigen::Vector4f(V.col(I(i)).x(), V.col(I(i)).y(), V.col(I(i)).z(), 1.0);
            Eigen::Vector4f transformedVertex2 = instance.mvp * Eigen::Vector4f(V.col(I(i+1)).x(), V.col(I(i+1)).y(), V.col(I(i+1)).z(), 1.0);
            Eigen::Vector4f transformedVertex3 = instance.mvp * Eigen::Vector4f(V.col(I(i+2)).x(), V.col(I(i+2)).y(), V.col(I(i+2)).z(), 1.0);
            if(ptInTriangle(xworld, yworld, transformedVertex1.x(), transformedVertex1.y(), transformedVertex2.x(), transformedVertex2.y(), transformedVertex3.x(), transformedVertex3.y())){
                selectedInstanceId = instance.id;
                enableCursorTrack = true;
                break;
            }
            i = i + 3;;
        }
        if(selectedInstanceId != -1){
            break;
        }
    }
}*/

bool findInstanceSelected(GLFWwindow *window){
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    
    GLuint index;
    glReadPixels(xpos, screen_height - ypos - 1, 1, 1, GL_STENCIL_INDEX, GL_UNSIGNED_INT, &index);
    if(index > 0){
        selectedInstanceId = index;
        return true;
    }
    return false;
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int mods)
{
    // Get the position of the mouse in the window
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);
    
    // Get the size of the window
    int width, height;
    glfwGetWindowSize(window, &width, &height);
    
    // Convert screen position to world coordinates
    Eigen::Vector4f p_screen(xpos,height-1-ypos,0,1); // NOTE: y axis is flipped in glfw
    Eigen::Vector4f p_canonical((p_screen[0]/width)*2-1,(p_screen[1]/height)*2-1,0,1);
    Eigen::Vector4f p_world = setTotalView && !totalView.isZero() ? totalView.inverse() * p_canonical : p_canonical;
    
    if(actionTriggered == Action::TRANSLATION){
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            switch (action)
            {
                case GLFW_PRESS:
                    no_of_clicks_translate = no_of_clicks_translate + 1;
                    switch (no_of_clicks_translate)
                {
                    case 1:
                    {
                        if(findInstanceSelected(window)){
                            pointer_x = p_world.x();
                            pointer_y = p_world.y();
                            enableCursorTrack = true;
                        }
                        break;
                    }
                    default:
                        break;
                }
                    break;
                case GLFW_RELEASE:
                    switch (no_of_clicks_translate)
                {
                    case 1:
                        enableCursorTrack = false;
                        break;
                    case 2:
                        updateChangesToSelectedInstance();
                        selectedInstanceId = -1;
                        no_of_clicks_translate = 0;
                        colorUpdated = false;
                    default:
                        break;
                }
                default:
                    break;
            }
        }
    }
}

void updateBaseMVP(){
    for(auto& instance : instanceCollection){
        instance.baseMVP =calculateBaseMVP(instance.baseModel);
    }
}

void window_size_callback(GLFWwindow* window, int width, int height)
{
    screen_width = width;
    screen_height = height;
    updateBaseMVP();
}

void updateTransformationToTheSelectedInstance(Transformation transform, string action){
    if(selectedInstanceId > 0){
        for(auto& instance: instanceCollection){
            if(instance.id == selectedInstanceId){
                Eigen::Matrix4f transformMatrix = Eigen::Matrix4f::Identity();
                switch (transform) {
                    case SCALE:
                        if(action == "UP"){
                            transformMatrix =  translate(target - instance.object.center) * scale(1.25) * translate(instance.object.center - target);
                        } else if(action == "DOWN"){
                            transformMatrix =  translate(target - instance.object.center) * scale(0.75) * translate(instance.object.center - target);
                        }
                        break;
                    case ROTATE:
                        if(action == "CW"){
                            transformMatrix =  translate(target - instance.object.center) * rotationAboutZ(10) * translate(instance.object.center - target);
                        } else if(action == "CCW"){
                            transformMatrix = translate(target - instance.object.center) * rotationAboutZ(-10) * translate(instance.object.center - target);
                        }
                        break;
                    default:
                        break;
                }
                instance.transformationMVP = transformMatrix * instance.transformationMVP;
                instance.transformationModel = transformMatrix * instance.transformationModel;
                break;
            }
        }
    }
}

void adjustCameraViewBy(Eigen::Vector3f adjustBy){
    cameraPosition = cameraPosition + adjustBy;
    if(I.size() > 0){
        for(auto& instance: instanceCollection){
            instance.baseMVP = calculateBaseMVP(instance.baseModel);
        }
    }
}

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(action == GLFW_RELEASE){
        // Update the position of the first vertex if the keys 1,2, or 3 are pressed
        switch (key)
        {
            case GLFW_KEY_I:
                actionTriggered = Action::INSERTION;
                break;
            case GLFW_KEY_O:
                actionTriggered = Action::TRANSLATION;
                selectedInstanceId = -1;
                colorUpdated = false;
                break;
            case GLFW_KEY_1:
                if(actionTriggered == Action::INSERTION){
                    addObjectToTheScene(ObjectName::UNIT_CUBE);
                } else if(actionTriggered == Action::TRANSLATION) {
                    updateColorToTheSelectedInstance(0);
                }
                break;
            case GLFW_KEY_2:
                if(actionTriggered == Action::INSERTION){
                    addObjectToTheScene(ObjectName::BUMPY_CUBE);
                } else if(actionTriggered == Action::TRANSLATION) {
                     updateColorToTheSelectedInstance(1);
                }
                break;
            case GLFW_KEY_3:
                if(actionTriggered == Action::INSERTION){
                    addObjectToTheScene(ObjectName::BUNNY);
                } else if(actionTriggered == Action::TRANSLATION) {
                      updateColorToTheSelectedInstance(2);
                }
                break;
            case GLFW_KEY_4:
                if(actionTriggered == Action::TRANSLATION) {
                      updateColorToTheSelectedInstance(3);;
                }
                break;
            case GLFW_KEY_5:
                if(actionTriggered == Action::TRANSLATION) {
                      updateColorToTheSelectedInstance(4);
                }
                break;
            case GLFW_KEY_6:
                if(actionTriggered == Action::TRANSLATION) {
                     updateColorToTheSelectedInstance(5);
                }
                break;
            case GLFW_KEY_7:
                if(actionTriggered == Action::TRANSLATION) {
                      updateColorToTheSelectedInstance(6);
                }
                break;
            case GLFW_KEY_8:
                if(actionTriggered == Action::TRANSLATION) {
                      updateColorToTheSelectedInstance(7);
                }
                break;
            case GLFW_KEY_9:
                if(actionTriggered == Action::TRANSLATION) {
                     updateColorToTheSelectedInstance(8);
                }
                break;
            case GLFW_KEY_W:
                rendering = RenderType::WIRE_FRAME;
                glUniform1i(program.uniform("shadingType"), rendering);
                break;
            case GLFW_KEY_F:
                rendering = RenderType::FLAT_SHADING;
                glUniform1i(program.uniform("shadingType"), rendering);
                break;
            case GLFW_KEY_P:
                rendering = RenderType::PHONG_SHADING;
                glUniform1i(program.uniform("shadingType"), rendering);
                break;
            case GLFW_KEY_Z:
                if(actionTriggered == Action::TRANSLATION) {
                    updateTransformationToTheSelectedInstance(Transformation::SCALE, "UP");
                }
                break;
            case GLFW_KEY_X:
                if(actionTriggered == Action::TRANSLATION) {
                    updateTransformationToTheSelectedInstance(Transformation::SCALE, "DOWN");
                }
            case GLFW_KEY_R:
                if(actionTriggered == Action::TRANSLATION) {
                    updateTransformationToTheSelectedInstance(Transformation::ROTATE, "CW");
                }
                break;
            case GLFW_KEY_T:
                if(actionTriggered == Action::TRANSLATION) {
                    updateTransformationToTheSelectedInstance(Transformation::ROTATE, "CCW");
                }
                break;
            case GLFW_KEY_LEFT:
                adjustCameraViewBy(Eigen::Vector3f(-1.0, 0.0, 0.0));
                break;
            case GLFW_KEY_RIGHT:
                adjustCameraViewBy(Eigen::Vector3f(1.0, 0.0, 0.0));
                break;
            case GLFW_KEY_UP:
                adjustCameraViewBy(Eigen::Vector3f(0.0, 1.0, 0.0));
                break;
            case GLFW_KEY_DOWN:
                adjustCameraViewBy(Eigen::Vector3f(0.0, -1.0, 0.0));
                break;
            case GLFW_KEY_K:
                projectionType = Projection::Perspective;
                updateBaseMVP();
                break;
            case GLFW_KEY_L:
                projectionType = Projection::Orthographic;
                updateBaseMVP();
                break;
            default:
                break;
        }
    }
}


int main(void)
{
    GLFWwindow *window;

    // Initialize the library
    if (!glfwInit())
        return -1;

    // Activate supersampling
    glfwWindowHint(GLFW_SAMPLES, 8);

    // Ensure that we get at least a 3.2 context
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);

    // On apple we have to load a core profile with forward compatibility
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(screen_width, screen_height, "Assignment3_All_tasks", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

#ifndef __APPLE__
    glewExperimental = true;
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
    }
    glGetError(); // pull and savely ignonre unhandled errors like GL_INVALID_ENUM
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
#endif

    int major, minor, rev;
    major = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MAJOR);
    minor = glfwGetWindowAttrib(window, GLFW_CONTEXT_VERSION_MINOR);
    rev = glfwGetWindowAttrib(window, GLFW_CONTEXT_REVISION);
    printf("OpenGL version recieved: %d.%d.%d\n", major, minor, rev);
    printf("Supported OpenGL is %s\n", (const char *)glGetString(GL_VERSION));
    printf("Supported GLSL is %s\n", (const char *)glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    cout << "******* All tasks are covered *******" << endl;

    // Initialize the VAO
    // A Vertex Array Object (or VAO) is an object that describes how the vertex
    // attributes are stored in a Vertex Buffer Object (or VBO). This means that
    // the VAO is not the actual object storing the vertex data,
    // but the descriptor of the vertex data.
    VertexArrayObject VAO;
    VAO.init();
    VAO.bind();

    // Initialize the VBO with the vertices data
    // A VBO is a data container that lives in the GPU memory
    VBO.init();
    //CBO.init();
    IBO.init();
    NBO.init();
    
    colorCodes.resize(3, 13);
    colorCodes <<
    0.0, 1.0, 1.0, 0.5, 0.0, 0.5, 0.75, 0.25, 0.75, 1.0, 0.0, 0.75, 0.0,
    1.0, 0.0, 1.0, 0.5, 0.5, 0.0, 0.75, 1.0,  1.0,  0.0, 1.0, 0.75, 0.0,
    1.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,  0.75, 0.25, 0.0, 0.0, 0.25, 1.0;

    // Initialize the OpenGL Program
    // A program controls the OpenGL pipeline and it must contains
    // at least a vertex shader and a fragment shader to be valid
    /*const GLchar *vertex_shader =
        "#version 150 core\n"
        "in vec3 position;"
        "in vec3 normal;"
        "uniform mat4 mvp;"
        "uniform mat4 model;"
        "out vec3 vNormal;"
        "out vec3 fragPos;"
        "void main()"
        "{"
        "   gl_Position = mvp * vec4(position, 1.0);"
        "   vNormal = mat3(transpose(inverse(model))) * normal;"
        "   fragPos = vec3(model * vec4(position, 1.0)); "
        "}";
    const GLchar *fragment_shader =
        "#version 150 core\n"
        "in vec3 vNormal;"
        "in vec3 fragPos;"
        "uniform vec3 objectColor;"
        "uniform vec3 lightPosition;"
        "uniform vec3 lightColor;"
        "uniform vec3 cameraPosition;"
        "uniform bool isPongShading;"
        "flat out vec4 outColor;"
        "void main()"
        "{"
        "float Ka = 0.3;"
        "float Kd = 0.2;"
        "float Ks = 0.5;"
        "vec3 ambient = Ka * lightColor;"
        "vec3 norm = normalize(vNormal);"
        "vec3 lightDir = normalize(lightPosition - fragPos);"
        "float diff = max(dot(norm, lightDir), 0.0);"
        "vec3 diffuse = Kd * diff * lightColor;"
        "vec3 specular = vec3(0.0, 0.0, 0.0);"
        "if(isPongShading){"
        "vec3 viewDir = normalize(cameraPosition - fragPos);"
        "vec3 reflectDir = reflect(-lightDir, norm);"
        "float spec = pow(max(dot(viewDir, reflectDir), 0.0), 16);"
        "specular = Ks * spec * lightColor; } "
        "vec3 fragColor = (ambient + diffuse + specular) * objectColor;"
        "outColor = vec4(fragColor, 1.0);"
        "}";*/
    const GLchar *vertex_shader =
    "#version 150 core\n"
    "in vec3 position;"
    "in vec3 normal;"
    "uniform mat4 mvp;"
    "uniform mat4 model;"
    "uniform vec3 objectColor;"
    "uniform vec3 lightPosition;"
    "uniform vec3 lightColor;"
    "uniform vec3 cameraPosition;"
    "uniform int shadingType;"
    "flat out vec3 flatColor;"
    "smooth out vec3 smoothColor;"
    "void main()"
    "{"
    "   gl_Position = mvp * vec4(position, 1.0);"
    "   vec3 vNormal = mat3(transpose(inverse(model))) * normal;"
    "   vec3 fragPos = vec3(model * vec4(position, 1.0)); "
    "   float Ka = 0.3;"
    "   float Kd = 0.2;"
    "   float Ks = 0.5;"
    "   vec3 ambient = Ka * lightColor;"
    "   vec3 norm = normalize(vNormal);"
    "   vec3 lightDir = normalize(lightPosition - fragPos);"
    "   float diff = max(dot(norm, lightDir), 0.0);"
    "   vec3 diffuse = Kd * diff * lightColor;"
    "   vec3 specular = vec3(0.0, 0.0, 0.0);"
    "   if(shadingType == 2){"
    "   vec3 viewDir = normalize(cameraPosition - fragPos);"
    "   vec3 reflectDir = reflect(-lightDir, norm);"
    "   float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);"
    "   specular = Ks * spec * lightColor; } "
    "   flatColor = (ambient + diffuse + specular) * objectColor;"
    "   smoothColor = (ambient + diffuse + specular) * objectColor;"
    "}";
    const GLchar *fragment_shader =
    "#version 150 core\n"
    "flat in vec3 flatColor;"
    "smooth in vec3 smoothColor;"
    "uniform int shadingType;"
    "out vec4 outColor;"
    "void main()"
    "{"
    "   if(shadingType == 1){"
    "       outColor = vec4(flatColor, 1.0);"
    "   } else {"
    "       outColor = vec4(smoothColor, 1.0);"
    "   }"
  
    "}";

    // Compile the two shaders and upload the binary to the GPU
    // Note that we have to explicitly specify that the output "slot" called outColor
    // is the one that we want in the fragment buffer (and thus on screen)
    program.init(vertex_shader, fragment_shader, "outColor");
    program.bind();

    // Register the keyboard callback
    glfwSetKeyCallback(window, key_callback);

    // Register the mouse callback
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    // Register the cursor position callback
    glfwSetCursorPosCallback(window, cursor_position_callback);
    
    // window resize callback
    glfwSetWindowSizeCallback(window, window_size_callback);
    
    glUniform3fv(program.uniform("lightPosition"), 1, LightSource::position.data());
    glUniform3fv(program.uniform("lightColor"), 1, LightSource::color.data());
    glUniform3fv(program.uniform("cameraPosition"), 1, cameraPosition.data());
    glUniform1i(program.uniform("shadingType"), RenderType::WIRE_FRAME);

    // Loop until the user closes the window
    while (!glfwWindowShouldClose(window))
    {
        // Bind your VAO (not necessary if you have only one)
        VAO.bind();

        // Bind your program
        program.bind();

        // Clear the framebuffer
        glClearColor(0.5f, 0.5f, 0.5f, 1.0f);
        // this is the default value(background) for stencil
        glClearStencil(0);
        //glClear(GL_COLOR_BUFFER_BIT);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        
        // Enable depth test
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
         // default cull face 'GL_BACK' and front face is 'GL_CCW'
        
        // Enable blend
        //glEnable(GL_BLEND);
        //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        
        /* Enable stencil operations */
        glEnable(GL_STENCIL_TEST);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        
        drawOutput();

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();
    }

    // Deallocate opengl memory
    program.free();
    VAO.free();
    VBO.free();
    //CBO.free();
    IBO.free();
    NBO.free();

    // Deallocate glfw internals
    glfwTerminate();
    return 0;
}
