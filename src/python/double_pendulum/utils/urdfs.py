import re
import xml.etree.ElementTree as ET
from lxml import etree


def remove_namespaces(tree):
    for el in tree.iter():
        match = re.match("^(?:\{.*?\})?(.*)$", el.tag)
        if match:
            el.tag = match.group(1)


def generate_urdf(
    urdf_in,
    urdf_out,
    mass=[0.5, 0.6],
    length=[0.3, 0.2],
    com=[0.3, 0.2],
    damping=[0.1, 0.1],
    coulomb_fric=[0.0, 0.0],
    inertia=[0.054, 0.025],
    torque_limit=[6.0, 6.0],
    model_pars=None,
):
    m = mass
    l = length
    r = com
    b = damping
    cf = coulomb_fric
    # g = gravity
    I = inertia
    tl = torque_limit

    if model_pars is not None:
        m = model_pars.m
        l = model_pars.l
        r = model_pars.r
        b = model_pars.b
        cf = model_pars.cf
        # g = model_pars.g
        I = model_pars.I
        # Ir = model_pars.Ir
        # gr = model_pars.gr
        tl = model_pars.tl

    # ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree = ET.parse(urdf_in, parser=etree.XMLParser())
    remove_namespaces(tree)
    root = tree.getroot()

    for joint in root.iter("joint"):
        if joint.attrib["name"] == "Joint1":
            for a in joint.iter("dynamics"):
                a.attrib["damping"] = str(b[0])
                a.attrib["friction"] = str(cf[0])
            for a in joint.iter("limit"):
                a.attrib["effort"] = str(tl[0])
            # for a in joint.iter('origin'):
            #     a.attrib['xyz'] = "0 "+str(length)+" 0"
        if joint.attrib["name"] == "Joint2":
            for a in joint.iter("dynamics"):
                a.attrib["damping"] = str(b[1])
                a.attrib["friction"] = str(cf[1])
            for a in joint.iter("limit"):
                a.attrib["effort"] = str(tl[1])
            for a in joint.iter("origin"):
                a.attrib["xyz"] = "0.0522 0 " + str(-l[0])

    for link in root.iter("link"):
        if link.attrib["name"] == "Link1":
            for a in link.iter("inertial"):
                for aa in a.iter("mass"):
                    aa.attrib["value"] = str(m[0])
                for aa in a.iter("origin"):
                    aa.attrib["xyz"] = "0 0 " + str(-r[0])
                for aa in a.iter("inertia"):
                    aa.attrib["ixx"] = str(I[0])
                    aa.attrib["iyy"] = str(I[0])
                    aa.attrib["izz"] = str(I[0])
            # for a in link.iter('visual'):
            #     for aa in a.iter('geometry'):
            #         for aaa in aa.iter('cylinder'):
            #             aaa.attrib['length'] = str(length)
        if link.attrib["name"] == "Link2":
            for a in link.iter("inertial"):
                for aa in a.iter("mass"):
                    aa.attrib["value"] = str(m[1])
                for aa in a.iter("origin"):
                    aa.attrib["xyz"] = "0 0 " + str(r[1])
                for aa in a.iter("inertia"):
                    aa.attrib["ixx"] = str(I[1])
                    aa.attrib["iyy"] = str(I[1])
                    aa.attrib["izz"] = str(I[1])

    tree.write(urdf_out)
