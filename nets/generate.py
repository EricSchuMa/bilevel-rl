import argparse
import numpy as np

route_str = """<routes>
\t<route id="r_ns" edges="npn_nt1 nt1_nps"/>
\t<route id="r_sn" edges="nps_nt1 nt1_npn"/>
\t<route id="r_we" edges="npw_nt1 nt1_npe"/>
\t<route id="r_ew" edges="npe_nt1 nt1_npw"/>
"""

flow_str = "\t<flow id=\"{id}\" route=\"{route}\" begin=\"{begin}\" end=\"{end}\" probability=" \
           "\"{prob}\" departPos=\"free\"/>\n "


def generate_stat_routefile(route_file, pwe=0.25, pns=0.25, num_seconds=3600):
    with open(route_file, 'w') as route:
        # routes
        route.write(route_str)
        write_flows(route, "0", 0, num_seconds, pns, pwe)
        route.write("</routes>")


def generate_dyn_routefile(route_file, pwe_peak=0.25, pns_peak=0.25, num_seconds=3600):
    ratios = [0.4, 0.7, 0.9, 1.0, 0.75, 0.5, 0.25, 0.7, 0.8, 0.9, 0.3, 0.1]

    with open(route_file, 'w') as route:
        # routes
        route.write(route_str)
        end_times = np.linspace(300, num_seconds, len(ratios), dtype=int)
        begin_times = end_times - 300

        for ratio, begin, end in zip(ratios, begin_times, end_times):
            write_flows(route, str(end), begin, end, pns_peak*ratio, pwe_peak*ratio)

        route.write("</routes>")


def write_flows(file, idx, begin, end, pns, pwe):
    flow_ns = flow_str.format(id="ns_" + idx, route="r_ns", begin=begin, end=end, prob=pns)
    flow_sn = flow_str.format(id="sn_" + idx, route="r_sn", begin=begin, end=end, prob=pns)
    flow_we = flow_str.format(id="we_" + idx, route="r_we", begin=begin, end=end, prob=pwe)
    flow_ew = flow_str.format(id="ew_" + idx, route="r_ew", begin=begin, end=end, prob=pwe)

    file.write(flow_ns)
    file.write(flow_sn)
    file.write(flow_we)
    file.write(flow_ew)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pns", dest="pns", type=float, default=0.2, required=False)
    parser.add_argument("--pwe", dest="pwe", type=float, default=0.2, required=False)
    parser.add_argument("--target-file",
                        dest="target_file",
                        type=str,
                        default="single_intersection/stat.gen.rou.xml",
                        required=False)
    parser.add_argument("--traffic-type", dest="traffic_type", type=str, default="static", required=False)

    args = parser.parse_args()

    if args.traffic_type == 'static':
        generate_stat_routefile(args.target_file, pns=args.pns, pwe=args.pwe)
    elif args.traffic_type == 'dynamic':
        generate_dyn_routefile(args.target_file, pwe_peak=args.pwe, pns_peak=args.pns)

    print(f"Generated routes with pns={args.pns} and pwe={args.pwe}.")
