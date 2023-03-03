import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torchview import draw_graph
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
from metrics.fid_score import calculate_fid


class Pix2Pix():
    def __init__(self, generator, discriminator, gen_loss, dis_loss):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        self.gen_loss = gen_loss
        self.dis_loss = dis_loss
        self.D_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.history = {
            "G_loss": [],
            "D_loss": [],
            "FID": []
        }

        self.load_model_weights()
        self.load_history()

    def init_weights(self, net, init_type='normal', scaling=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv')) != -1:
                nn.init.normal_(m.weight.data, 0.0, scaling)
            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, scaling)
                nn.init.constant_(m.bias.data, 0.0)

        print("Initialize network with %s" % init_type)
        net.apply(init_func)

    def get_generator_loss(self, generated_image, target_img, G, real_target):
        gen_loss = self.dis_loss(G, real_target)
        l1_l = self.gen_loss(generated_image, target_img)
        gen_total_loss = gen_loss + (100 * l1_l)
        return gen_total_loss

    def get_discriminator_loss(self, output, label):
        disc_loss = self.dis_loss(output, label)
        return disc_loss
    
    def train_step(self, input_img, target_img, imgs_dir, batch_size):
        self.D_optimizer.zero_grad()
        input_img = input_img.to(self.device)
        target_img = target_img.to(self.device)
 
        # ground truth labels real and fake
        real_target = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(self.device))
        fake_target = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(self.device))

        # generator forward pass
        generated_image = self.generator(input_img)
         
        # train discriminator with fake/generated images
        disc_inp_fake = torch.cat((input_img, generated_image), 1)
         
        D_fake = self.discriminator(disc_inp_fake.detach())
         
        D_fake_loss = self.get_discriminator_loss(D_fake, fake_target)
         
        # train discriminator with real images
        disc_inp_real = torch.cat((input_img, target_img), 1)
                                 
        D_real = self.discriminator(disc_inp_real)
        D_real_loss = self.get_discriminator_loss(D_real,  real_target)
 
     
         
        # average discriminator loss
        D_total_loss = (D_real_loss + D_fake_loss) / 2
        # compute gradients and run optimizer step
        D_total_loss.backward()
        self.D_optimizer.step()
         
         
        # Train generator with real labels
        self.G_optimizer.zero_grad()
        fake_gen = torch.cat((input_img, generated_image), 1)
        G = self.discriminator(fake_gen)
        G_loss = self.get_generator_loss(generated_image, target_img, G, real_target)                                 
        # compute gradients and run optimizer step
        G_loss.backward()
        self.G_optimizer.step()

        return {
            "G_loss": G_loss.item(),
            "D_loss": D_total_loss.item(),
        }

    def print_result(self, epoch, num_epochs):
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFID: %.4f'
              % (epoch, num_epochs,
                 self.history["D_loss"][-1], self.history["G_loss"][-1], self.history["FID"][-1]))

    def save_model(self):
        if os.path.exists("./weights/generator_weights.pth"):
            os.remove("./weights/generator_weights.pth")

        if os.path.exists("./weights/discriminator_weights.pth"):
            os.remove("./weights/discriminator_weights.pth")

        torch.save(self.generator.state_dict(), "./weights/generator_weights.pth")
        torch.save(self.discriminator.state_dict(), "./weights/discriminator_weights.pth")

        print("==> Model saved.")

    def load_model_weights(self):
        if os.path.exists("./weights/generator_weights.pth"):
            self.generator.load_state_dict(torch.load("./weights/generator_weights.pth"))
        else:
            self.init_weights(self.generator, 'normal', scaling=0.02)
        
        if os.path.exists("./weights/discriminator_weights.pth"):
            self.discriminator.load_state_dict(torch.load("./weights/discriminator_weights.pth"))
        else:
            self.init_weights(self.discriminator, 'normal', scaling=0.02)

        print("==> Model loaded.")

    def save_history(self):
        fig, ax = plt.subplots()
        ax.plot(self.history["G_loss"], label="G_loss")
        ax.plot(self.history["D_loss"], label="D_loss")
        ax.legend()
        fig.savefig('./history.png')

        fig, ax = plt.subplots()
        ax.plot(self.history["FID"], label="FID")
        ax.legend()
        fig.savefig('./FID_score.png')

        with open("./cache/history.pickle", 'wb') as file:
            pickle.dump(self.history, file)
        print("==> History saved.")

    def load_history(self):
        if os.path.exists("./cache/history.pickle"):
            with open("./cache/history.pickle", 'rb') as file:
                self.history = pickle.load(file)

        print("==> History loaded.")

    def train_pix2pix(self, train_dataloader, test_dataloader, epochs, imgs_dir, batch_size):
        self.generator.train()
        self.discriminator.train()

        min_FID = None

        for epoch in range(1, epochs+1):

            print(f'Epoch {epoch}')

            # Train data
            print('Train')
            for (input_img, target_img) in tqdm(train_dataloader):
                res = self.train_step(input_img=input_img, target_img=target_img, imgs_dir=imgs_dir, batch_size=batch_size)

                self.history["G_loss"].append(res['G_loss'])
                self.history["D_loss"].append(res['D_loss'])
   
            # Calculate FID
            print("Calulate FID")
            gen_imgs = []
            
            for (input_img, target_img) in tqdm(test_dataloader):
                gen_img = self.generator(input_img.to(self.device)).cpu().detach()
                gen_imgs.append(gen_img)

            gen_imgs = torch.cat(gen_imgs, dim=0)
            fid_score = calculate_fid(imgs_dir, gen_imgs, batch_size, self.device)

            self.history["FID"].append(fid_score)

            # Print result
            self.print_result(epoch, epochs)
            
            # Save model
            if (min_FID is None) or min_FID > self.history["FID"][-1] :
                min_FID = self.history["FID"][-1]
                self.save_model()
   
            # Save history
            self.save_history()

    def save_structure_model(self, batch_size, resize):
        # generator
        input = torch.randn(batch_size, 3, resize, resize, device=self.device)
        model_graph = draw_graph(self.generator, input_data=input)
        dot = model_graph.visual_graph
        dot.format = 'png'
        dot.filename = "Generator"
        dot.render(directory="./model_structure").replace('\\', '/')

        # discriminator
        input = torch.randn(batch_size, 6, resize, resize, device=self.device)
        model_graph = draw_graph(self.discriminator, input_data=input)
        dot = model_graph.visual_graph
        dot.format = 'png'
        dot.filename = "Discriminator"
        dot.render(directory="./model_structure").replace('\\', '/')

