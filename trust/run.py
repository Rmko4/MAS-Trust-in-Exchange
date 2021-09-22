from trust.model import PDTModel

model = PDTModel(N=1000, neighbourhood_size=50, mobility_rate=0.2)
# print([a.neighbourhood for a in model.schedule.agents])
model.run_model()