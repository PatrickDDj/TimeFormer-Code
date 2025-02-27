import torch

def main():
    a = torch.zeros(3)

    a = a.cuda()

    while True:
        loss = a.sum()
        loss.backward()
    

if __name__ == "__main__":
    main()
