import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import multiprocessing as mp

from Module import monotonic_classifier as mc
from Module.mappings import equiv_to_key, equiv_to_case



def cr_models(p, df):
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    tr1 = df[p1].values.tolist()
    tr2 = df[p2].values.tolist()
    diag = df['target'].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

    models = mc.compute_recursion(data, None,(rev, up, key))

    return p1, p2, models, data




def get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y):
    if key == 1:
        return {"xy": (min_x, min_y), "width": abs(x - min_x), "height": abs(y - min_y)}
    elif key == 2:
        return {"xy": (x, min_y), "width": max_x + abs(x), "height": abs(y - min_y)}
    elif key == 3:
        return {"xy": (x, y), "width": max_x + abs(x), "height": max_y + abs(y)}
    else:
        return {"xy": (min_x, y), "width": abs(x - min_x), "height": max_y + abs(y)}

### WITH THE GREY AREA

def print_model(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.set_facecolor("lightgray")

        reg_err, bpr, bpb, r_p, b_p = models[key]

        x_r, y_r, x_b, y_b = [], [], [], []

        for xy, w, lab in data:
            x, y = xy
            if lab == 0:  # blue
                x_r.append(x)
                y_r.append(y)
            else:  # red
                x_b.append(x)
                y_b.append(y)

        min_x = min(min(x_r), min(x_b)) - 0.05
        min_y = min(min(y_r), min(y_b)) - 0.05
        max_x = max(max(x_r), max(x_b)) + 0.05
        max_y = max(max(y_r), max(y_b)) + 0.05

        # Plot blue rectangles
        for bp in bpb:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightsteelblue', zorder=1))

        # Plot red rectangles
        for bp in bpr:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightcoral', zorder=1))

        random.shuffle(data)

        for d in data:
            if d[2] == 0:
                plt.scatter(d[0][0], d[0][1], c = 'royalblue', marker='.', zorder = 2)
            elif d[2] == 1:
                plt.scatter(d[0][0], d[0][1], c = 'firebrick', marker='.', zorder = 2)

        plt.xlabel(p1)
        plt.ylabel(p2)

        if cm is not None:
            pair = '/'.join([p1, p2, str(key)])
            error = cm.at['LOOCVE', pair]
            plt.title('CE = {} & LOOCVE = {}'.format(round(reg_err / len(data), 3), round(error, 3)))
        else:
            plt.title('CE = {}'.format(round(reg_err / len(data), 3)))

        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png')
        else:
            plt.show()
            
            
def show_results(df, pairs, nbcpus, pathname, cm):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )
    pool.close()

    for r in res:
        p1, p2, models, data = r
        print_model(data, models, p1, p2, pathname, cm)

            
            
#### FAVORING ONE CLASS

def print_model_fav_class0(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.set_facecolor("lightsteelblue")

        reg_err, bpr, bpb, r_p, b_p = models[key]

        x_r, y_r, x_b, y_b = [], [], [], []

        for xy, w, lab in data:
            x, y = xy
            if lab == 0:  # blue
                x_r.append(x)
                y_r.append(y)
            else:  # red
                x_b.append(x)
                y_b.append(y)

        min_x = min(min(x_r), min(x_b)) - 0.05
        min_y = min(min(y_r), min(y_b)) - 0.05
        max_x = max(max(x_r), max(x_b)) + 0.05
        max_y = max(max(y_r), max(y_b)) + 0.05


        # Plot red rectangles
        for bp in bpr:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightcoral', zorder=1))

        random.shuffle(data)

        for d in data:
            if d[2] == 0:
                plt.scatter(d[0][0], d[0][1], c = 'royalblue', marker='.', zorder = 2)
            elif d[2] == 1:
                plt.scatter(d[0][0], d[0][1], c = 'firebrick', marker='.', zorder = 2)

        plt.xlabel(p1)
        plt.ylabel(p2)

        if cm is not None:
            pair = '/'.join([p1, p2, str(key)])
            error = cm.at['LOOCVE', pair]
            plt.title('CE = {} & LOOCVE = {}'.format(round(reg_err / len(data), 3), round(error, 3)))
        else:
            plt.title('CE = {}'.format(round(reg_err / len(data), 3)))

        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png')
        else:
            plt.show()

def print_model_fav_class1(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.set_facecolor("lightcoral")

        reg_err, bpr, bpb, r_p, b_p = models[key]

        x_r, y_r, x_b, y_b = [], [], [], []

        for xy, w, lab in data:
            x, y = xy
            if lab == 0:  # blue
                x_r.append(x)
                y_r.append(y)
            else:  # red
                x_b.append(x)
                y_b.append(y)

        min_x = min(min(x_r), min(x_b)) - 0.05
        min_y = min(min(y_r), min(y_b)) - 0.05
        max_x = max(max(x_r), max(x_b)) + 0.05
        max_y = max(max(y_r), max(y_b)) + 0.05

        # Plot blue rectangles
        for bp in bpb:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightsteelblue', zorder=1))

        random.shuffle(data)

        for d in data:
            if d[2] == 0:
                plt.scatter(d[0][0], d[0][1], c = 'royalblue', marker='.', zorder = 2)
            elif d[2] == 1:
                plt.scatter(d[0][0], d[0][1], c = 'firebrick', marker='.', zorder = 2)

        plt.xlabel(p1)
        plt.ylabel(p2)

        if cm is not None:
            pair = '/'.join([p1, p2, str(key)])
            error = cm.at['LOOCVE', pair]
            plt.title('CE = {} & LOOCVE = {}'.format(round(reg_err / len(data), 3), round(error, 3)))
        else:
            plt.title('CE = {}'.format(round(reg_err / len(data), 3)))

        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png')
        else:
            plt.show()


def show_results_class1(df, pairs, nbcpus, pathname, cm):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )
    pool.close()

    for r in res:
        p1, p2, models, data = r
        print_model_fav_class1(data, models, p1, p2, pathname, cm)


def show_results_class0(df, pairs, nbcpus, pathname, cm):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )
    pool.close()

    for r in res:
        p1, p2, models, data = r
        print_model_fav_class0(data, models, p1, p2, pathname, cm)
        
### WITH A NEW POINT

def print_model_out_fav_class1(data, out, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.set_facecolor("lightcoral")

        reg_err, bpr, bpb, r_p, b_p = models[key]

        x_r, y_r, x_b, y_b = [], [], [], []

        for xy, w, lab in data:
            x, y = xy
            if lab == 0:  # blue
                x_r.append(x)
                y_r.append(y)
            else:  # red
                x_b.append(x)
                y_b.append(y)

        min_x = min(min(x_r), min(x_b)) - 0.05
        min_y = min(min(y_r), min(y_b)) - 0.05
        max_x = max(max(x_r), max(x_b)) + 0.05
        max_y = max(max(y_r), max(y_b)) + 0.05

        # Plot blue rectangles
        for bp in bpb:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightsteelblue', zorder=1))

        random.shuffle(data)

        for d in data:
            if d[2] == 0:
                plt.scatter(d[0][0], d[0][1], c = 'royalblue', marker='.', zorder = 2)
            elif d[2] == 1:
                plt.scatter(d[0][0], d[0][1], c = 'firebrick', marker='.', zorder = 2)

        plt.xlabel(p1)
        plt.ylabel(p2)
        
        plt.scatter(out[0],out[1],c = 'green', marker='.', zorder = 2 )

        if cm is not None:
            pair = '/'.join([p1, p2, str(key)])
            error = cm.at['LOOCVE', pair]
            plt.title('CE = {} & LOOCVE = {}'.format(round(reg_err / len(data), 3), round(error, 3)))
        else:
            plt.title('CE = {}'.format(round(reg_err / len(data), 3)))

        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png')
        else:
            plt.show()





def print_model_out(data, out, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(3, 3))
        ax = plt.axes()
        ax.set_facecolor("lightgray")

        reg_err, bpr, bpb, r_p, b_p = models[key]

        x_r, y_r, x_b, y_b = [], [], [], []

        for xy, w, lab in data:
            x, y = xy
            if lab == 0:  # blue
                x_r.append(x)
                y_r.append(y)
            else:  # red
                x_b.append(x)
                y_b.append(y)

        min_x = min(min(x_r), min(x_b)) - 0.05
        min_y = min(min(y_r), min(y_b)) - 0.05
        max_x = max(max(x_r), max(x_b)) + 0.05
        max_y = max(max(y_r), max(y_b)) + 0.05

        # Plot blue rectangles
        for bp in bpb:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightsteelblue', zorder=1))

        # Plot red rectangles
        for bp in bpr:
            x, y = bp
            rect_params = get_rectangle_params(key, x, y, min_x, min_y, max_x, max_y)
            ax.add_artist(patches.Rectangle(**rect_params, facecolor='lightcoral', zorder=1))

        random.shuffle(data)

        for d in data:
            if d[2] == 0:
                plt.scatter(d[0][0], d[0][1], c = 'royalblue', marker='.', zorder = 2)
            elif d[2] == 1:
                plt.scatter(d[0][0], d[0][1], c = 'firebrick', marker='.', zorder = 2)
                
        plt.scatter(out[0],out[1],c = 'green', marker='.', zorder = 2 )

        plt.xlabel(p1)
        plt.ylabel(p2)

        if cm is not None:
            pair = '/'.join([p1, p2, str(key)])
            error = cm.at['LOOCVE', pair]
            plt.title('CE = {} & LOOCVE = {}'.format(round(reg_err / len(data), 3), round(error, 3)))
        else:
            plt.title('CE = {}'.format(round(reg_err / len(data), 3)))

        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png')
        else:
            plt.show()








        
### IN THE RANKING SPACE

def show_results_RS(df, probs_df, pairs, nbcpus, pathname, cm):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )
    pool.close()

    for r in res:
        p1, p2, models, data = r
        print_model_RS(data, models, p1, p2, probs_df, pathname, cm)