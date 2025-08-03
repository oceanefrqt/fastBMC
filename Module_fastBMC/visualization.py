import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import multiprocessing as mp
import numpy as np

from Module_fastBMC import monotonic_classifier as mc
from Module_fastBMC.mappings import equiv_to_key, equiv_to_case

def split_label(str_label, max_length=20):
    words = str_label.split()
    result = []

    current_line = ''
    for word in words:
        if len(current_line) + len(word) > max_length:
            result.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += ' '
            current_line += word

    if current_line:
        result.append(current_line)

    return '\n'.join(result)


def cr_models(p, df):
    p1, p2, key = p.split('/')
    key = int(key)
    rev, up = equiv_to_key[key]
    tr1 = df[p1].values.tolist()
    tr2 = df[p2].values.tolist()
    diag = df['target'].values.tolist()
    data = [((tr1[n], tr2[n] ), 1, diag[n]) for n in range(len(diag))]

    models = mc.compute_recursion(data, (rev, up, key))

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

def print_model_uncertainty(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)
        
        cmap = plt.get_cmap('RdYlBu_r')


        low_color = cmap(0)
        high_color = cmap(256)
        
        colors = [low_color, high_color]

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
                
        step = 0.01

        minX = min(min(x_r), min(x_b))
        minY = min(min(y_r), min(y_b))
        maxX = max(max(x_r), max(x_b))
        maxY = max(max(y_r), max(y_b))

        X_ = np.arange(minX - step, maxX + step, step)
        Y_ = np.arange(minY - step, maxY + step, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.ones((len(Y_), len(X_)))*0.5 
        

        for bp in bpb:
            x, y = bp
            x = round(x,2)
            y = round(y,2)
                


            if key == 3:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = 0
                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0.5:
                                            Z[l,m] = 0
            elif key == 1:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0
                            for l in range(j, len(Y_)):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] =0

            elif key == 2:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0

                            for l in range(j, len(Y_)):
                                for m in range(i, len(X_)):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] = 0

            elif key == 4:
                 for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0

                            for l in range(0,j+1):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] =0
                                        
                                        
        for bp in bpr:
            x, y = bp
            x = round(x,2)
            y = round(y,2)
                


            if key == 1:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = 1
                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0.5:
                                            Z[l,m] = 1
            elif key == 3:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1
                            for l in range(j, len(Y_)):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] =1

            elif key == 4:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1

                            for l in range(j, len(Y_)):
                                for m in range(i, len(X_)):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] = 1

            elif key == 2:
                 for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1

                            for l in range(0,j+1):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0.5:
                                        Z[l,m] =1

        
        
        
        plt.figure(figsize=(3,3))

        p1 = split_label(p1)
        p2 = split_label(p2)

        plt.xlabel(p1, fontsize=16)
        plt.ylabel(p2, fontsize=16)

        if cm is not None:
            MAE = cm.at['MAE-CVE', p]
            fig.title(f"MAE = {round(MAE,2)}",fontsize=20)


        plt.contourf(sorted(X_),sorted(Y_),np.flipud(Z), alpha=0.5, cmap='RdYlBu_r')

        for d in data:
            plt.scatter(d[0][0], d[0][1], c=colors[int(d[2])])


        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png', bbox_inches = 'tight')
        else:
            plt.show()

            
            
def show_results(df, pairs, nbcpus, pathname, cm):
    pool = mp.Pool(nbcpus)
    vals = [(p, df) for p in pairs]

    res = pool.starmap(cr_models, vals, max(1,len(vals)//nbcpus) )
    pool.close()

    for r in res:
        p1, p2, models, data = r
        print_model_uncertainty(data, models, p1, p2, pathname, cm)

            
            
#### FAVORING ONE CLASS

def print_model_class0(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)
        
        cmap = plt.get_cmap('RdYlBu_r')


        low_color = cmap(0)
        high_color = cmap(256)
        
        colors = [low_color, high_color]

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
                
        step = 0.01

        minX = min(min(x_r), min(x_b))
        minY = min(min(y_r), min(y_b))
        maxX = max(max(x_r), max(x_b))
        maxY = max(max(y_r), max(y_b))

        X_ = np.arange(minX - step, maxX + step, step)
        Y_ = np.arange(minY - step, maxY + step, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.zeros((len(Y_), len(X_)))
        

                                        
                                        
        for bp in bpr:
            x, y = bp
            x = round(x,2)
            y = round(y,2)
                


            if key == 1:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = 1
                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 0:
                                            Z[l,m] = 1
            elif key == 3:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1
                            for l in range(j, len(Y_)):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0:
                                        Z[l,m] =1

            elif key == 4:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1

                            for l in range(j, len(Y_)):
                                for m in range(i, len(X_)):
                                    if Z[l,m] == 0:
                                        Z[l,m] = 1

            elif key == 2:
                 for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 1

                            for l in range(0,j+1):
                                for m in range(0, i+1):
                                    if Z[l,m] == 0:
                                        Z[l,m] =1

        
        
        
        plt.figure(figsize=(3,3))

        p1 = split_label(p1)
        p2 = split_label(p2)

        plt.xlabel(p1, fontsize=16)
        plt.ylabel(p2, fontsize=16)

        if cm is not None:
            MAE = cm.at['MAE-CVE', p]
            fig.title(f"MAE = {round(MAE,2)}",fontsize=20)


        plt.contourf(sorted(X_),sorted(Y_),np.flipud(Z), alpha=0.5, cmap='RdYlBu_r')

        for d in data:
            plt.scatter(d[0][0], d[0][1], c=colors[int(d[2])])


        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png', bbox_inches = 'tight')
        else:
            plt.show()


def print_model_fav_class1(data, models, p1, p2, pathname=None, cm=None):
    for key in models.keys():
        key = int(key)
        
        cmap = plt.get_cmap('RdYlBu_r')


        low_color = cmap(0)
        high_color = cmap(256)
        
        colors = [low_color, high_color]

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
                
        step = 0.01

        minX = min(min(x_r), min(x_b))
        minY = min(min(y_r), min(y_b))
        maxX = max(max(x_r), max(x_b))
        maxY = max(max(y_r), max(y_b))

        X_ = np.arange(minX - step, maxX + step, step)
        Y_ = np.arange(minY - step, maxY + step, step)


        Xv, Yv = np.meshgrid(sorted(X_), sorted(Y_, reverse=True))
        Z = np.ones((len(Y_), len(X_))) 
        

        for bp in bpb:
            x, y = bp
            x = round(x,2)
            y = round(y,2)
                


            if key == 3:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                                Z[j,i] = 0
                                for l in range(0,j+1):
                                    for m in range(i, len(X_)):
                                        if Z[l,m] == 1:
                                            Z[l,m] = 0
            elif key == 1:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0
                            for l in range(j, len(Y_)):
                                for m in range(0, i+1):
                                    if Z[l,m] == 1:
                                        Z[l,m] =0

            elif key == 2:
                for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0

                            for l in range(j, len(Y_)):
                                for m in range(i, len(X_)):
                                    if Z[l,m] == 1:
                                        Z[l,m] = 0

            elif key == 4:
                 for i in range(len(X_)):
                    for j in range(len(Y_)):
                        if x== round(Xv[j,i],2) and y == round(Yv[j,i],2):
                            Z[j,i] = 0

                            for l in range(0,j+1):
                                for m in range(0, i+1):
                                    if Z[l,m] == 1:
                                        Z[l,m] =0

        
        
        
        plt.figure(figsize=(3,3))

        p1 = split_label(p1)
        p2 = split_label(p2)

        plt.xlabel(p1, fontsize=16)
        plt.ylabel(p2, fontsize=16)

        if cm is not None:
            MAE = cm.at['MAE-CVE', p]
            fig.title(f"MAE = {round(MAE,2)}",fontsize=20)


        plt.contourf(sorted(X_),sorted(Y_),np.flipud(Z), alpha=0.5, cmap='RdYlBu_r')

        for d in data:
            plt.scatter(d[0][0], d[0][1], c=colors[int(d[2])])


        if pathname is not None:
            plt.savefig(pathname + p1 + '_' + p2 + '.png', bbox_inches = 'tight')
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

def print_model_out(data, out, models, p1, p2, df1, pathname = None):

    for key in models.keys():
        key = int(key)

        plt.figure(figsize=(5,5))
        ax = plt.axes()
        ax.set_facecolor("lightgray")
        
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

        if pathname is not None:
            plt.savefig(pathname + g1 + '_' + g2  + '.png')

            f = open(pathname + 'gene.txt', 'a')
            f.write('{} : {}\n'.format(g1, p1))
            f.write('{} : {}\n'.format(g2, p2))
            f.close()
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