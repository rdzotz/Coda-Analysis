#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:19:57 2018

@author: rwilson
"""


import pygmsh
import meshio
import numpy as np
import pickle

class utilities():
    '''A collection of functions for interacting with the mesh object
    '''

    def meshOjfromDisk(meshObjectPath='cly.Mesh'):
        '''Read the entire mesh object from disk.

        Parameters
        ----------
        meshObjectPath : str (default='cly.Mesh')
        '''
        with open(meshObjectPath, 'rb') as clyMesh_file:
            return pickle.load(clyMesh_file)


class mesher():
    '''Mesh generator class using pygmsh to gmsh code.

    Parameters
    ----------
    mesh_param : dict
        Expected parameters defining the mesh, char_len, height, radius
    cell_data : dict
        Contains line, tetra, triangle, vertex 'gmsh:physical' and
        'gmsh:geometrical'.
    cells : dict
        Contains line, tetra, triangle, vertex of the point indicies as defined in
        ``points``.
    points : array(float)
        Matrix of xyz coords for each point in the mesh domain

    Notes
    -----
     Understanding the mesh structure
         Points are a list of each point or verticies in x,y,z positions.
         cell_data['tetra']['gmsh:physical'] : the physical values of each  tetra
         * cell['tetra'] : list of lists of each tetrahedral verticies index referencing to
          the coords inside the points.                                [points[i1],
                                                                        points[i2],
                                                                        points[i3],
                                                                        points[i4]]
    '''

    def __init__(self, mesh_param):
        self.mesh_param = mesh_param
        self.cell_data = None
        self.points = None
        self.cells = None
        self.cell_cent = None

    def meshIt(self):
        '''Produces the mesh.
        '''

        self._cylinderMesh()

        self._cellCent()

    def _cylinderMesh(self):
        ''' Produce a cylindrical mesh
        '''

        # The geometry object
        geom = pygmsh.opencascade.Geometry()


        # Positions
        btm_face = [0.0, 0.0, 0.0]
        axis = [0.0, 0.0, self.mesh_param['height']]

        # create the cylinder with open cascade
        geom.add_cylinder(btm_face, axis, self.mesh_param['radius'],
            char_length=self.mesh_param['char_len']
            )

        # Make the mesh
        self.points, self.cells, _, self.cell_data, _ = pygmsh.generate_mesh(geom)



    def _cellCent(self):
        ''' Calculate the centre of each tetra.
        '''

        # The verticies in cart coords
        tetra_verts = [ np.array([self.points[vert[0]], self.points[vert[1]],
                                  self.points[vert[2]], self.points[vert[3]]])
                                     for vert in self.cells['tetra']]

        # The centre of tetra in cart coords
        self.cell_cent = [np.array([vert[:,0].sum()/4, vert[:,1].sum()/4, vert[:,2].sum()/4])
                                for vert in tetra_verts]

    def saveMesh(self, name):
        '''Save the mesh to file.

        Parameters
        ----------
        name : str
            Name of the mesh file saved to the current directory.
        '''

        meshio.write('%s.vtu' % name, self.points, self.cells, cell_data=self.cell_data)
        # meshio.write('%s.msh4' % name, self.points, self.cells, cell_data=self.cell_data)
        # meshio.gmsh_io.write('%s.msh' % name, self.points, self.cells, cell_data=self.cell_data)

    def setCellsVal(self, cell_values):
        '''Set each cell physical value.

        Parameters
        ----------
        cell_values : array/list
            physical values of each tetra cell within the mesh domain in order
            corresponding to ``points``.
        '''

        self.cell_data['tetra']['gmsh:physical'] = cell_values

    def meshOjtoDisk(self):
        '''Save the entire mesh object to disk
        '''
        with open('cly.Mesh', 'wb') as clyMesh_file:
          pickle.dump(self, clyMesh_file)

    def meshOjfromDisk(self):
        '''Save the entire mesh object to disk
        TODO
        ----
            Should likely depreciate this function and simply use that stored in the utility class
        '''
        with open('cly.Mesh', 'rb') as clyMesh_file:
            return pickle.load(clyMesh_file)

