from aiohttp import web
import aiohttp_cors
import numpy as np
from PIL import Image
import PIL.ImageOps
import io
import os
import sys
import glob
import sqlite3
import random as rn
import yaml
import coloredlogs
import pandas as pd
import fnmatch
import re
import logging
import asyncio
import requests
import ssl
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

coloredlogs.install(level=logging.INFO)
MAX_WORKERS = None
DBFILES = 'dbfiles'


def remove_prefix(text, prefix):
    return text[text.startswith(prefix) and len(prefix):]

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

def get_tile(x, y, z, inv, idx):
    byteIO = io.BytesIO()
    try:
        if x < 0:
            raise
        if y < 0:
            raise
        new_im = Image.new("RGB", (TILESIZE, TILESIZE), "#dddddd")
        itiles = int(NTILES / (2 ** z))
        resize = int(TILESIZE / itiles)
        x_off = 0
        for ix in range(itiles):
            y_off = 0
            for iy in range(itiles):
                ic = idx[iy + itiles * y][ix + itiles * x]
                if ic == -1:
                    continue
                if os.path.basename(images[ic]) in blacklist:
                    im = Image.new("RGB", (resize, resize), "#dddddd")
                else:
                    im = Image.open(images[ic])
                    im = im.resize((resize, resize))
                    if inv == 1:
                        if np.shape(im)[-1] == 3:
                            im = PIL.ImageOps.invert(im)
                        else:
                            im = PIL.ImageOps.invert(im.convert("RGB"))
                new_im.paste(im, (x_off, y_off))
                y_off += resize
            x_off += resize
        new_im.save(byteIO, "PNG")
        return byteIO.getvalue()
    except Exception as e:
        logging.debug(e)
        return None


async def update(request):
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    c.execute('SELECT updates from CONFIG')
    updates = c.fetchone()[0]
    if updates == 0:
        logging.info("No UPDATES Allowed")
        response = web.Response(text="", status=200)
        conn.commit()
        conn.close()
        return response
    gid = request.query["gid"]
    class_input = int(request.query["class"])
    c.execute("UPDATE IMAGES SET display = 0")
    c.execute("DELETE FROM COORDS")
    c.execute("UPDATE IMAGES SET display = 1 where class = {}".format(class_input))
    c.execute("SELECT id FROM IMAGES where display = 1 order by id")
    all_ids = c.fetchall()
    all_filtered = [i[0] for i in all_ids]
    if len(all_filtered) > 0:
        default_nx = int(np.ceil(np.sqrt(len(all_filtered))))
        NX = default_nx
        NY = default_nx
        update_config(NX, NY, len(all_filtered), userdb)
        NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
        logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = all_filtered
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images filtered and displayed".format(len(image_idx)))
    response = web.Response(text="", status=200)
    conn.commit()
    conn.close()
    return response


async def info(request):
    global idx
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    x = int(request.query["x"])
    y = int(request.query["y"])
    ic = idx[y][x]
    c.execute("SELECT name,hostName,class,SNname, NED_redshift, NED_vel, NED_type, raMean, decMean, rPSFMag, r_i, g_r, i_z, z_y, TransientRedshift, TransientRA, TransientDEC, TransientDiscoveryDate, TransientDiscoveryMag, rApRadius, rKronRad, gMag, rMag, iMag, zMag, yMag FROM META where id = {} order by id".format(ic))
    spec = {}
    specHost = {}
    lcSN = {}
    try:
        name, hostName, classid, SNname, hostZ, NED_v, NED_type, raMean, decMean, rPSFMag, r_i, g_r, i_z, z_y, transientZ, transientRA, transientDEC, transientDiscDate, transientDiscMag, ApRadius, rKronRad, gMag, rMag, iMag, zMag, yMag = c.fetchone()
        img = images[ic];
        print("static/pngs/" + img.split("/")[10]);
        logging.info("Selected: x={}, y={}, ic={}".format(x, y, ic))
        st = 200
        for file in os.listdir('/Users/alexgagliano/Documents/Research/Transient_ML/tables/SNspectra'):
            if fnmatch.fnmatch(file, '*%s*'% SNname):
                if file.startswith("osc_"):
                    spectra = pd.read_csv("/Users/alexgagliano/Documents/Research/Transient_ML/tables/SNspectra/" + file, header=None, names=['x', 'y', 'z'])
                else:
                    spectra = pd.read_csv("/Users/alexgagliano/Documents/Research/Transient_ML/tables/SNspectra/" + file, delim_whitespace=True, header=None, names=['x', 'y', 'z'])
                spectra = spectra.astype('float')
                spec = spectra.to_json()
        tempspecHost = find_all(remove_prefix(SNname, "SN") + "_hostSpectra.csv", '/Users/alexgagliano/Documents/Research/Transient_ML_Box/hostSpectra/')
        print(remove_prefix(SNname, "SN") + "_hostSpectra.csv")
        if tempspecHost:
            tempspecHost = pd.read_csv('/Users/alexgagliano/Documents/Research/Transient_ML_Box/hostSpectra/' + remove_prefix(SNname, "SN") + "_hostSpectra.csv")
            tempspecHost = tempspecHost.astype("float")
            tempspecHost.rename(index={0:'x',1:'y',2:'z'})
            specHost = tempspecHost.to_json()
        import glob 
        lcSN_name = glob.glob("/Users/alexgagliano/Documents/Research/Transient_ML_Box/SNphotometry/*%s*" % remove_prefix(SNname, "SN"))
        if lcSN_name:   
            lcSN = pd.read_csv(lcSN_name[0])
            lcSN = lcSN.to_json()
            print(lcSN_name[0])
    except Exception as e:
        logging.debug(e)
        logging.info(e)
        st = 404
        ic = -1
        img = "";
        classid = -1
        hostName = ""
        name = ""
        SNname = ""
        hostZ = -1
        NED_v = -1
        NED_type = ""
        raMean = -1
        decMean = -1
        rPSFMag = -1
        r_i = -1
        g_r = -1
        i_z = -1
        z_y = -1
        transientZ = -1
        transientRA = -1
        transientDEC = -1
        ApRadius = -1
        rKronRad = -1
        gMag = -1
        rMag = -1
        iMag = -1
        zMag = -1
        yMag = -1
        transientDiscDate = ""
        transientDiscMag = -1
    if ic >= 0:
        logging.info("  Galaxy: {}, SN Name: {}, SN Class: {}".format(hostName, SNname, classid))
    else:
        name = ""

    print(lcSN)
    response = web.json_response({"name":name, "lcSN":lcSN, "specHost":specHost, "spec":spec, "raMean":raMean, "decMean": decMean, "transientRA":transientRA, "transientDEC":transientDEC, "hostZ":hostZ, "transientZ":transientZ, "z_y":z_y, "r_i":r_i, "g_r":g_r, "i_z":i_z, "NED_v":NED_v, "NED_type":NED_type, "ApRadius":ApRadius, "KronRadius":rKronRad, "gMag":gMag, "rMag":rMag, "iMag":iMag, "zMag":zMag, "yMag":yMag,"rPSFMag":rPSFMag, "transientDiscDate":transientDiscDate, "transientDiscMag":transientDiscMag, "SNname": SNname, "hostName": hostName, "class": classid, "status": st, "img": img})
    conn.commit()
    conn.close()
    return response

async def infoall(request):
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    c.execute("SELECT a.id,a.name,a.class,a.display,c.SNname, c.NED_redshift, c.NED_vel, c.NED_type, c.raMean, c.decMean, c.rPSFMag, c.r_i, c.g_r, c.i_z, c.z_y, c.TransientRedshift, c.TransientRA, c.TransientDEC, c.rApRadius, c.rKronRad, c.gMag, c.rMag, c.iMag, c.zMag, c.yMag, c.TransientDiscoveryDate, c.TransientDiscoveryMag, b.vx,b.vy FROM IMAGES a, COORDS b, META c where a.class >= 0 and a.display = 1 and a.id = b.id and a.name = c.name order by a.id")
    allreturn = c.fetchall()
    logging.info("Getting info for all {} classified tiles".format(len(allreturn)))
    st = 200
    names = [i[1] for i in allreturn]
    classids = [i[2] for i in allreturn]
    SNnames = [i[4] for i in allreturn]
    hostZ = [i[5] for i in allreturn]
    NED_v = [i[6] for i in allreturn]
    NED_type = [i[7] for i in allreturn]
    raMean = [i[8] for i in allreturn]
    decMean = [i[9] for i in allreturn]
    rPSFMag = [i[10] for i in allreturn]
    r_i = [i[11] for i in allreturn]
    g_r = [i[12] for i in allreturn]
    i_z = [i[13] for i in allreturn]
    z_y = [i[14] for i in allreturn]
    transientZ = [i[15] for i in allreturn]
    transientRA = [i[16] for i in allreturn]
    transientDEC = [i[17] for i in allreturn]
    all_ApRadius = [i[18] for i in allreturn]
    all_KronRad = [i[19] for i in allreturn]
    all_gMag = [i[20] for i in allreturn]
    all_rMag = [i[21] for i in allreturn]
    all_iMag = [i[22] for i in allreturn]
    all_zMag = [i[23] for i in allreturn]
    all_yMag = [i[24] for i in allreturn]
    all_tDate = [i[25] for i in allreturn]
    all_tMag = [i[26] for i in allreturn]
    vx = [i[27] for i in allreturn]
    vy = [i[28] for i in allreturn]
    response = web.json_response(
            {"SNnames":SNnames, "names": names, "classes": classids, "vx": vx, "vy": vy, "status": st, "hostZ" : hostZ, "NED_v": NED_v, "NED_type": NED_type, "raMean": raMean, "decMean":decMean,
                "rPSFMag": rPSFMag, "r_i":r_i, "g_r":g_r, "i_z": i_z, "z_y": z_y, "transientZ":transientZ, "transientRA":transientRA, "transientDEC":transientDEC, "ApRadius":all_ApRadius, "KronRadius":all_KronRad, "gMag":all_gMag, "rMag":all_rMag, "iMag":all_iMag, "zMag":all_zMag, "yMag":all_yMag, "transientDiscDate":all_tDate, "transientDiscMag":all_tMag})
    conn.commit()
    conn.close()
    return response

async def query(request):
    global idx, NX, NY, NTILES
    logging.info("QUERY: ")
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    qq = request.query["query"]
    #qq = "SELECT name from META where META.name = '00526.png'"
    qq = "SELECT name from META where (META.SNname='SN{}') or (META.SNname='{}') or (META.classLabels='SN {}') or (META.classLabels='SN {}') or (META.classLabels='{}') or (META.hostName='{}')".format(remove_prefix(request.query["query"], "SN "), request.query["query"], request.query["query"], remove_prefix(request.query["query"], "SN"), request.query["query"], request.query["query"])
    logging.info("{}".format(qq))
    long_qq = 'select IMAGES.id from IMAGES, ({}) F where IMAGES.name = F.name and IMAGES.display = 1 and IMAGES.class != 0 order by id;'.format(qq)
    try:
        c.execute(long_qq)
        all_ids = c.fetchall()
    except Exception as e:
        logging.error(str(e))
        conn.commit()
        conn.close()
        if qq.strip() == '':
            msg = "Empty query"
        else:
            msg = str(e) 
        response = web.json_response({"msg": msg,  "status": "400"})
        return response
    all_filtered = [i[0] for i in all_ids]
    if len(all_filtered) > 0:
        default_nx = int(np.ceil(np.sqrt(len(all_filtered))))
        NX = default_nx
        NY = default_nx
        update_config(NX, NY, len(all_filtered), userdb)
        NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
        logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = all_filtered
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images filtered and displayed".format(len(image_idx)))
    response = web.json_response({"msg": "ok",  "status": "200"})
    return response

async def filter(request):
    global idx, NX, NY, NTILES
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    logging.info("FILTER: ")
    checked = request.query["checked"]
    checked = checked.split(",")[:-1]
    logging.info(checked)
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    if len(checked) == 0:
        c.execute("SELECT id FROM IMAGES where display = 1 and class != 0 order by id")
    else:
        c.execute(
            "SELECT id FROM IMAGES where display = 1 and class in ({}) order by id".format(
                ",".join(checked)
            )
        )
    all_ids = c.fetchall()
    all_filtered = [i[0] for i in all_ids]
    if len(all_filtered) > 0:
        if len(checked) == 0:
            images0, total_images, nimages, dbname, NX, NY, NTILES, MAXZOOM, TILESIZE, config = read_config(
                "config.yaml", n_images=len(all_filtered)
            )
            del images0
            update_config(NX, NY, len(all_filtered), userdb)
        else:
            default_nx = int(np.ceil(np.sqrt(len(all_filtered))))
            NX = default_nx
            NY = default_nx
            update_config(NX, NY, len(all_filtered), userdb)
        NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
        logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = all_filtered
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images filtered and displayed".format(len(image_idx)))
    response = web.Response(text="", status=200)
    return response


async def random(request):
    global idx, NX, NY, NTILES
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    logging.info("RANDOMIZE: ")
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    c.execute("SELECT id FROM IMAGES where display = 1 and class != 0 order by id")
    all_ids = c.fetchall()
    all_displayed = [i[0] for i in all_ids]
    nim = rn.randint(10, nimages)
    default_nx = int(np.ceil(np.sqrt(nim)))
    NX = default_nx
    NY = default_nx
    update_config(NX, NY, nim, userdb)
    NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
    logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = rn.sample(all_displayed, nim)
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images displayed".format(nim))
    response = web.Response(text="", status=200)
    return response


async def reset(request):
    global idx, blacklist, images
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    logging.info("RESET: ")
    images, total_images, nimages, dbname, NX, NY, NTILES, MAXZOOM, TILESIZE, config = read_config(
        "config.yaml", force_copy=True
    )
    idx, blacklist = initialize(images, nimages, NX, NY, NTILES, userdb)
    update_config(NX, NY, nimages, userdb)
    # initiate_db(dbname, images)
    response = web.Response(text="", status=200)
    return response


async def input_query(request):
    global idx, NX, NY, NTILES
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    c.execute(request)
    all_ids = c.fetchall()
    all_displayed = [i[0] for i in all_ids]
    images0, total_images, nimages, dbname, NX, NY, NTILES, MAXZOOM, TILESIZE, config = read_config(
        "config.yaml", n_images=len(all_displayed)
    )
    del images0
    update_config(NX, NY, len(all_displayed), userdb)
    NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
    logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = all_displayed
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images displayed".format(len(all_displayed)))
    response = web.Response(text="", status=200)
    return response



async def redraw(request):
    global idx, NX, NY, NTILES
    userdb = os.path.join(DBFILES, request.query["user"] + '.db')
    logging.info("REDRAW: ")
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    c.execute("SELECT id FROM IMAGES where display = 1 and class != -1 order by id")
    all_ids = c.fetchall()
    all_displayed = [i[0] for i in all_ids]
    images0, total_images, nimages, dbname, NX, NY, NTILES, MAXZOOM, TILESIZE, config = read_config(
        "config.yaml", n_images=len(all_displayed)
    )
    del images0
    update_config(NX, NY, len(all_displayed), userdb)
    NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
    logging.info("NX x NY = {} x {}. NTILES = {}".format(NX, NY, NTILES))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = all_displayed
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    logging.info(" {} images displayed".format(len(all_displayed)))
    response = web.Response(text="", status=200)
    return response


async def sort(request):
    global idx
    print("SORT: ")
    image_idx = np.argsort(df_final.COLOR1.values)
    temp = np.zeros(NX * NY, dtype="int") - 1
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    response = web.Response(text="", status=200)
    return response


async def main(request):
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    x = int(request.query["x"])
    y = int(request.query["y"])
    z = int(request.query["z"])
    inv = int(request.query["inv"])
    logging.debug("{}, {}, {}, {}".format(x, y, z, inv))
    loop = asyncio.get_running_loop()
    #tile = await request.app.loop.run_in_executor(executor, get_tile, x, y, z, inv, idx)
    tile = await loop.run_in_executor(executor, get_tile, x, y, z, inv, idx)
    if tile is not None:
        response = web.Response(body=tile, status=200, content_type="image/png")
    else:
        response = web.Response(text="", status="404")
    return response


async def init_db(request):
    global idx, blacklist
    username = request.query["user"]
    userdb = os.path.join(DBFILES, username + '.db')
    if not os.path.exists(userdb):
        logging.info("Creating DB for {}".format(username))
        create_db(userdb)
        initiate_db(userdb, images)
        idx = None
    if idx is None:
        logging.info("Creating idx for the first time")
        idx, blacklist = initialize(images, nimages, NX, NY, NTILES, userdb)
        merge_db(userdb)
    response = web.Response(text="", status="200")
    return response


def update_config(NX, NY, nimages, userdb, updates=None):
    conn = sqlite3.connect(userdb)
    c = conn.cursor()
    if updates is None:
        c.execute('SELECT updates from CONFIG')
        updates = c.fetchone()[0]
    chunk = [0, nimages, NX, NY, updates]
    c.execute("INSERT or REPLACE INTO CONFIG VALUES (?,?,?,?,?)", chunk)

    conn.commit()
    conn.close()

def read_config_single(conf, arg1, arg2):
    with open(conf, "r") as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    return config[arg1][arg2]

def read_config(conf, force_copy=False, n_images=None, no_read_images=False):
    with open(conf, "r") as cfg:
         config = yaml.load(cfg, Loader=yaml.FullLoader)
    if no_read_images:
        images = []
        total_images = 1e9
    else:
        images = sorted(glob.glob(os.path.join(config["display"]["path"], "*.png")))
        total_images = len(images)
    if total_images == 0:
        logging.error("Images not found!. Please check path")
        sys.exit(0)
    dbname = config["display"]["dataname"] + ".db"
    logging.info("Total Images in path: {}".format(total_images))
    nimages = int(config["display"]["nimages"])
    if n_images is not None:
        nimages = n_images
    if nimages <= 0:
        logging.error("nimages must be positive")
        sys.exit(0)
    if nimages > total_images:
        logging.warning(
            "nimages({}) > total_images({}). Defaulting to total_images".format(
                nimages, total_images
            )
        )
        nimages = total_images
    logging.info("No Images displayed: {}".format(nimages))
    NX = config["display"]["xdim"]
    NY = config["display"]["ydim"]
    if NX * NY < nimages:
        default_nx = int(np.ceil(np.sqrt(nimages)))
        logging.warning(
            "xdim by ydim ({0}) do not match the nimages({1}).Defaulting to square {2} x {2}".format(
                NX * NY, nimages, default_nx
            )
        )
        NX = default_nx
        NY = default_nx
    config["display"]["xdim"] = NX
    config["display"]["ydim"] = NY
    config["display"]["nimages"] = nimages
    outyaml = config["display"]["dataname"] + ".yaml"
    if force_copy:
        with open(outyaml, "w") as out:
            out.write(yaml.dump(config))
    NTILES = int(2 ** np.ceil(np.log2(max(NX, NY))))
    MAXZOOM = np.log2(NTILES)
    DELTAZOOM = config["display"]["deltaZoom"]
    MINZOOM = max(0, MAXZOOM - DELTAZOOM)
    TILESIZE = config["display"]["tileSize"]
    logging.info(
        "{} max tiles in one side, maxzoom = {}, minzoom = {}, maxsize = {} pxs".format(
            NTILES, MAXZOOM, MINZOOM, NTILES * TILESIZE
        )
    )
    logging.info("Input array NX x NY = {} x {}".format(NX, NY))
    logging.info("Actual array NX x NY = {} x {}".format(NX, int(np.ceil(nimages / NX))))
    return (
        images,
        total_images,
        nimages,
        dbname,
        NX,
        NY,
        NTILES,
        MAXZOOM,
        TILESIZE,
        config,
    )


def create_db(filedb):
    print("Creating custom db file");
    conn = sqlite3.connect(filedb)
    c = conn.cursor()
    c.execute(
        "create table if not exists IMAGES "
        "(id int primary key, display int default 1, name text, class int default -1)"
    )
    c.execute(
        "create table if not exists COORDS " "(id int primary key, vx int, vy int)"
    )
    c.execute(
        "create table if not exists CONFIG " "(id int primary key, nimages int , nx int, ny int, updates int default 1)"
    )
    conn.commit()
    conn.close()


def initiate_db(filedb, images):
    chunk = [(i, 0, os.path.basename(images[i]), -1) for i in range(len(images))]
    conn = sqlite3.connect(filedb)
    c = conn.cursor()
    c.executemany("INSERT or REPLACE INTO IMAGES VALUES (?,?,?,?)", chunk)
    conn.commit()
    conn.close()

def merge_db(dbname):
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='{}';".format('META'))
    results = c.fetchall()
    if len(results) == 0:
        logging.info("MERGE: Table META does not exist")
    else:
        logging.info("MERGE: Merging tables")
        c.execute("""UPDATE IMAGES
                        SET
                        class = ( select META.class from META  where META.name = IMAGES.name)
                        WHERE
                        EXISTS (
                        SELECT *
                        FROM META
                        WHERE META.name = IMAGES.name)""")
    conn.commit()
    conn.close()

def initialize(images, nimages, NX, NY, NTILES, dbname):
    logging.info('Initializing {}'.format(dbname))
    temp = np.zeros(NX * NY, dtype="int") - 1
    image_idx = rn.sample(range(len(images)), nimages)  # np.arange(nimages)
    conn = sqlite3.connect(dbname)
    c = conn.cursor()
    c.execute("UPDATE IMAGES SET display = 0")
    c.execute("DELETE FROM COORDS")
    for i in image_idx:
        c.execute("UPDATE IMAGES SET display = 1 where id = {}".format(i))
    c.execute("SELECT id FROM IMAGES where display = 1 order by id")
    all_ids = c.fetchall()
    all_displayed = [i[0] for i in all_ids]
    image_idx = all_displayed
    temp[: len(image_idx)] = image_idx
    temp = temp.reshape((NY, NX))
    idx = np.pad(
        temp, ((0, NTILES - NY), (0, NTILES - NX)), "constant", constant_values=-1
    )
    blacklist = []
    for i in image_idx:
        vy, vx = np.where(idx == i)
        c.execute("INSERT INTO COORDS VALUES ({},{},{})".format(i, vx[0], vy[0]))
    conn.commit()
    conn.close()
    updates = read_config_single('config.yaml', 'operation', 'updates')
    if updates:
        update_config(NX, NY, nimages, dbname, 1)
    else:
        update_config(NX, NY, nimages, dbname, 0)
    return idx, blacklist

if __name__ == "__main__":
    global idx, images, NX, NY, NTILES
    if not os.path.exists(DBFILES):
        os.makedirs(DBFILES)
    if not os.path.exists("config.yaml"):
        logging.error("config.yaml not found. Use config_template.yaml as example")
        logging.error("Exiting")
        sys.exit()
    logging.basicConfig(level=logging.INFO)
    images, total_images, nimages, dbname, NX, NY, NTILES, MAXZOOM, TILESIZE, config = read_config(
        "config.yaml", force_copy=True
    )
    with open("config.yaml", "r") as cfg:
        configT = yaml.load(cfg, Loader=yaml.FullLoader)
    if configT["server"]["ssl"]:
        sslname = configT["server"]["sslName"]
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(
            "ssl/{}.crt".format(sslname), "ssl/{}.key".format(sslname)
        )
    idx = None
    blacklist = []
    # create_db(dbname)
    # initiate_db(dbname, images)
    # idx, blacklist = initialize(images, nimages, NX, NY, NTILES, dbname)
    # Web app
    app = web.Application()
    cors = aiohttp_cors.setup(app)
    cors = aiohttp_cors.setup(
        app,
        defaults={
            # Allow all to read all CORS-enabled resources from client
            "*": aiohttp_cors.ResourceOptions()
        },
    )
    host = config["server"]["host"]
    port = config["server"]["port"]
    root = config["server"]["rootUrl"]
    rnn = cors.add(app.router.add_resource("{rootUrl}random".format(rootUrl=root)))
    inf = cors.add(app.router.add_resource("{rootUrl}info".format(rootUrl=root)))
    upd = cors.add(app.router.add_resource("{rootUrl}update".format(rootUrl=root)))
    get = cors.add(app.router.add_resource("{rootUrl}getall".format(rootUrl=root)))
    sor = cors.add(app.router.add_resource("{rootUrl}sort".format(rootUrl=root)))
    fil = cors.add(app.router.add_resource("{rootUrl}filter".format(rootUrl=root)))
    qry = cors.add(app.router.add_resource("{rootUrl}query".format(rootUrl=root)))
    res = cors.add(app.router.add_resource("{rootUrl}reset".format(rootUrl=root)))
    red = cors.add(app.router.add_resource("{rootUrl}redraw".format(rootUrl=root)))
    ini = cors.add(app.router.add_resource("{rootUrl}initdb".format(rootUrl=root)))
    que = cors.add(app.router.add_resource("{rootUrl}input_query".format(rootUrl=root)))
    cors.add(rnn.add_route("GET", random))
    cors.add(inf.add_route("GET", info))
    cors.add(get.add_route("GET", infoall))
    cors.add(upd.add_route("GET", update))
    cors.add(sor.add_route("GET", sort))
    cors.add(fil.add_route("GET", filter))
    cors.add(qry.add_route("GET", query))
    cors.add(res.add_route("GET", reset))
    cors.add(red.add_route("GET", redraw))
    cors.add(que.add_route("GET", input_query))
    cors.add(ini.add_route("GET", init_db))
    app.router.add_routes([web.get("{rootUrl}".format(rootUrl=root), main)])
    logging.info("======== Running on {}:{}{} ========".format(host, port, root))
    logging.info("(Press CTRL+C to quit)")
    if configT["server"]["ssl"]:
        web.run_app(
            app, port=config["server"]["port"], print=None, access_log=None, ssl_context=ssl_context
        )
    else:
        web.run_app(app, port=config["server"]["port"], access_log=None, print=None)
