from cls_models.steel_classifier_inference import run, parse_opt


def main():
    opt = parse_opt()
    run(**vars(opt))

if __name__ == '__main__':
    main()