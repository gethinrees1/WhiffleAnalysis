# This version stored the best trees in a min-heap to keep the top N best trees.
# bad_pairs_cog8.py checks for clashes withi the minimum encroachment distance
import argparse
import ast
import csv
import heapq
import itertools
import re

import numpy as np

# Global store of part properties and bad pairs
properties = {}
bad_pairs = set()
direction = "up"
myunits = "imperial"  #"metric"
myfactor = 25.4 if myunits == "metric" else 1.0
minlen = 4 * myfactor #minimum length of link
# Note: myfactor is set based on units (25.4 for metric, 1 for imperial) and is passed at run time
min_encroachment= 1*myfactor  # min allowable distance to avoid encroachment
minrat = 4 #minimum ratio allowable in link length caclulation


# Minimum length for moment-based rejection (in mm)
MAX_ACCEPTABLE_MOMENT = 1.0e9
# max number of best trees to keep
TOP_N = 20


def quote_labels(tree_str):
    return re.sub(r'\b([A-Za-z]+)\b', r"'\1'", tree_str)

# Recursively flatten the tree into merge steps
def flatten_merge_steps(tree):
    steps = []

    def recurse(node):
        if isinstance(node, str):
            return node
        left = recurse(node[0]) #
        right = recurse(node[1])
        merged = left + right
        steps.append((left, right, merged))
        return merged

    recurse(tree)
    return steps

def format_merge_line(tree):
    steps = flatten_merge_steps(tree)
    parts = [f"{left};{right};{merged}; ;" for left, right, merged in steps]
    return ''.join(parts)


def try_add_tree(best_trees, tree_string, moment_score, average_moment, worst_pair, total_distance, max_length):
    # Prepare the full record (you can add more info here)
    record = (moment_score, tree_string, average_moment, worst_pair, total_distance, max_length)
    
    if len(best_trees) < TOP_N:
        heapq.heappush(best_trees, record)
    else:
        # Replace worst in heap if this is better
        heapq.heappushpop(best_trees, record)

def detect_n_from_properties(filename):
    labels = set()
    with open(filename, newline='') as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(csvfile, dialect=dialect)
        for row in reader:
            labels.add(row['Label'])
    return len(labels)

def compute_moment(mass, cg_obj, cg_merged):
    dx = cg_obj[0] - cg_merged[0]
    dy = cg_obj[1] - cg_merged[1]
    dz = cg_obj[2] - cg_merged[2]
    return mass * ((dx**2 + dy**2 + dz**2)**0.5)


def load_properties(filename):
    with open(filename, newline='') as csvfile:
        sample = csvfile.read(1024)
        csvfile.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
        reader = csv.DictReader(csvfile, dialect=dialect)
        for row in reader:
            label = row['Label']
            mass = float(row['Mass'])
            x = float(row['X'])
            y = float(row['Y'])
            z = float(row['Z'])
            if direction in ("up", "down"):
                z = 0.0
            elif direction in ("fwd", "aft"):
                x = 0.0
            else:
                y = 0.0
            cg = (x, y, z)
            properties[label] = {'mass': mass, 'cg': cg}

def check_pair(a, b, current_labels):
    pair = tuple(sorted([a, b]))
    if pair in bad_pairs:
        return False

    # Retrieve property data
    if a not in properties or b not in properties:
        print(f"Missing data for {a} or {b}")
        return False

    prop_a = properties[a]
    prop_b = properties[b]

    # Example physical check: CG distance must be under 10 units
    dx = prop_a['cg'][0] - prop_b['cg'][0]
    dy = prop_a['cg'][1] - prop_b['cg'][1]
    dz = prop_a['cg'][2] - prop_b['cg'][2]
    distance = (dx**2 + dy**2 + dz**2)**0.5

    ratio = max(prop_a['mass'],prop_b['mass'])/min(prop_a['mass'],prop_b['mass'])

    if (ratio>minrat) or (distance/(ratio+1) < minlen):
        bad_pairs.add(pair)
        log_bad_pair(pair)
        return False

    # --- New check: no other point within 25.4mm of the new merge point ---
    # Compute new CG for AB
    m1, m2 = prop_a['mass'], prop_b['mass']
    cg1, cg2 = prop_a['cg'], prop_b['cg']
    total_mass = m1 + m2
    new_cg = tuple((m1 * c1 + m2 * c2) / total_mass for c1, c2 in zip(cg1, cg2))

    # Check all other points in current_labels (except a and b)
    for label in current_labels:
        if label in (a, b):
            continue
        cg_other = properties[label]['cg']
        d = sum((c1 - c2) ** 2 for c1, c2 in zip(new_cg, cg_other)) ** 0.5
        if d < 1*myfactor: # myfacor is 25.4 for metric, 1 for imperial
            # Too close to another point
            return False

    return True

def log_bad_pair(pair):
    with open("bad_pairs_log.txt", "a") as log:
        log.write(f"{pair[0]},{pair[1]}\n")

def merge_properties(label1, label2):
    prop1 = properties[label1]
    prop2 = properties[label2]
    m1, m2 = prop1['mass'], prop2['mass']
    cg1, cg2 = prop1['cg'], prop2['cg']

    total_mass = m1 + m2
    new_cg = tuple((m1 * c1 + m2 * c2) / total_mass for c1, c2 in zip(cg1, cg2))
    diff_rss = sum(d**2 for d in tuple(c2 - c1 for c1, c2 in zip(cg1, cg2))) ** 0.5  # Root sum square of diff
    len1=sum(d**2 for d in tuple(c1 - new_cg for c1, new_cg in zip(cg1, new_cg))) ** 0.5  # Root sum square of diff
    len2=sum(d**2 for d in tuple(c2 - new_cg for c2, new_cg in zip(cg2, new_cg))) ** 0.5  # Root sum square of diff
    longest_len = max(len1, len2)
    new_label = ''.join(sorted([label1, label2]))
    properties[new_label] = {'mass': total_mass, 'cg': new_cg}

    # Compute moments
    mmt1 = compute_moment(m1, cg1, new_cg)
    mmt2 = compute_moment(m2, cg2, new_cg)
    worst_moment = max(mmt1, mmt2)
    worst_pair = f"{label1},{label2}"   # maintain unsorted pair for traceability

    return new_label, worst_moment, worst_pair,diff_rss,longest_len


def reduce_tree(tree_str):
    pattern = re.compile(r'\(([A-Z]+),([A-Z]+)\)') #find pairs of labels in parentheses
    max_moment = 0.0
    worst_pair = None
    total_distance = 0.0
    max_length = 0.0
    average_moment = 0.0
    total_moment = 0.0
    count = 0

    # Track all original points
    original_labels = set(re.findall(r'[A-Z]', tree_str))
    original_cgs = {label: properties[label]['cg'] for label in original_labels}

    # Track all merges: (new_label, new_cg, merged_labels)
    merges = []

    current_labels = set(original_labels)

    while True:
        match = pattern.search(tree_str)
        if not match:
            break

        a, b = match.group(1), match.group(2)
        if not check_pair(a, b, current_labels):
            return None, None, None, None, None, None

        new_label, moment, pair, dist, longest_len = merge_properties(a, b)

        # Track the merge
        merges.append((new_label, properties[new_label]['cg'], {a, b}))

        # Update current_labels
        current_labels.remove(a)
        current_labels.remove(b)
        current_labels.add(new_label)

        total_moment += moment
        count += 1

        if moment > max_moment:
            max_moment = moment
            worst_pair = pair

        if longest_len > max_length:
            max_length = longest_len
        total_distance += dist

        tree_str = tree_str[:match.start()] + new_label + tree_str[match.end():]

    # --- Post-check: for each merge, ensure new CG is not within 25.4mm of any original point (except merged) ---
    for new_label, new_cg, merged_set in merges:
        for label1, label2 in itertools.combinations(original_labels, 2):
            if label1 in merged_set or label2 in merged_set:
                continue
            A = original_cgs[label1]
            B = original_cgs[label2]
            #dist = point_to_segment_distance(new_cg, A, B)
            #if dist < min_encroachment:
            #    return None, None, None, None, None, None

    # Optional: reject if max moment too high
    if MAX_ACCEPTABLE_MOMENT is not None and max_moment > MAX_ACCEPTABLE_MOMENT:
        return None, None, None, None, None, None

    average_moment = total_moment / count if count > 0 else 0.0
    return tree_str, max_moment, average_moment, worst_pair, total_distance, max_length


def process_tree_file(input_file, output_file):
    min_moment = float('inf')
    min_distance = float('inf')
    best_trees = []  # this will be our min-heap
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        outfile.write("OriginalTree,MaxMoment,AverageMoment,WorstPair,TotalDiatance,MaxLength\n")  # CSV header
        for line in infile:
            tree = line.strip()
            reduced, max_moment, average_moment, worst_pair, total_distance, max_length = reduce_tree(tree)
            # (moment_score, data_string) tuples
            if reduced is not None:
                write_line = False
                try_add_tree(best_trees, tree, max_moment, average_moment, worst_pair, total_distance, max_length)
                #if max_moment < min_moment:
                #    min_moment = max_moment
                #    write_line = True
                #if total_distance < min_distance:
                #    min_distance = total_distance
                #    write_line = True
                #if write_line:
        best_trees_sorted = sorted(best_trees)
        #record = (moment_score, tree_string, average_moment, worst_pair, total_distance, max_length)
        for score, tree, average, worst, total, maxl in best_trees_sorted:
            tree_str = quote_labels(tree)
            mytree = ast.literal_eval(tree_str)
            merged_line = format_merge_line(mytree)
            #outfile.write(f"{tree};{score:.3f};{average:.3f};{worst};{total:.3f};{maxl:.3f}\n")
            outfile.write(f"{merged_line},{score:.3f},{average:.3f},{worst};{total:.3f},{maxl:.3f}\n")
    print(f"Finished. Valid trees with moments written to {output_file}")


def point_to_segment_distance(P, A, B):
    # P, A, B are (x, y, z) tuples
    from math import sqrt
    # Vector from A to B
    AB = tuple(b - a for a, b in zip(A, B))
    # Vector from A to P
    AP = tuple(p - a for a, p in zip(A, P))
    # Dot products
    AB_AB = sum(x * x for x in AB)
    AB_AP = sum(a * b for a, b in zip(AB, AP))
    if AB_AB == 0:
        # A and B are the same point
        return sqrt(sum((p - a) ** 2 for p, a in zip(P, A)))
    t = max(0, min(1, AB_AP / AB_AB))
    closest = tuple(a + t * ab for a, ab in zip(A, AB))
    return sqrt(sum((p - c) ** 2 for p, c in zip(P, closest)))


# Example usage
if __name__ == "__main__":
   
    props_file = "real_cog.txt"
    n = detect_n_from_properties(props_file)
    load_properties(props_file)  # Load masses and CGs
    input_trees_file = f"trees_deduped{n}.txt"
    output_trees_file = f"reduced_trees{n}{direction}.txt"
    process_tree_file(input_trees_file, output_trees_file)
