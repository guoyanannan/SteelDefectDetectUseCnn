from db_process.info_post_process import parse_opt,run


def main():
    opt = parse_opt()
    run(**vars(opt))


if __name__ == '__main__':
    main()